# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch QSA math matching the transformers eager indexer.

Selection (per query, causal over logical positions ``0 .. p``):

- split visible tokens into complete blocks of ``compress_ratio`` plus a tail
- pool each complete block by a float mean, Gemma-RMSNorm, RoPE at the block
  start
- score ``relu(q · k).sum(heads) / sqrt(head_dim)`` and keep
  ``indexer_budget / compress_ratio`` blocks
- always keep the incomplete tail

Contexts of length ``<= indexer_budget`` therefore attend every causal token.
The index tensor is fixed width ``budget + compress_ratio - 1``, padded with
``-1``. Attention ignores those pads, so token order does not change the
softmax.
"""

from __future__ import annotations

import math
import os

import torch

from ._materialize import is_hpu_lazy_mode as _is_hpu_lazy_mode
from ._materialize import materialized as _materialized
from .control import positions_rows_equal as _positions_rows_equal
from .h23 import h23_enabled as _h23_enabled
from .h28 import h28_enabled as _h28_enabled

__all__ = [
    "apply_partial_rope",
    "clamp_qsa_indices",
    "dense_prefix_attention",
    "effective_qsa_width",
    "gemma_rmsnorm",
    "hpu_rms_norm",
    "is_hpu_lazy_mode",
    "mrope_cos_sin",
    "pool_indexer_blocks",
    "qsa_attention_path",
    "qsa_static_bucket_enabled",
    "select_qsa_indices",
    "select_qsa_indices_from_pooled",
    "sparse_attention",
]


def is_hpu_lazy_mode() -> bool:
    """True when Habana lazy execution is requested (PT_HPU_LAZY_MODE=1)."""
    return os.environ.get("PT_HPU_LAZY_MODE", "").strip().lower() in (
        "1", "true", "yes", "on")


def qsa_static_bucket_enabled(device_type: str | None = None) -> bool:
    """Static-capacity KV attention for lazy decode.

    ``VLLM_QWEN4EXP_QSA_STATIC_BUCKET``: ``0``/``false`` disables,
    ``1``/``true`` forces (any device), unset/``auto`` (default) enables
    only on HPU under lazy mode. In eager mode a ``[0:S_kv]`` slice has
    no recompile cost, so the exact active-prefix path is kept; in lazy
    mode a slice whose length grows every token (257, 258, ...) would
    hand Synapse a new recipe per token, whereas attending over the
    512-token backing capacity (masking positions >= S_kv) only changes
    shape when the backing buffer itself grows.
    """
    pref = os.environ.get("VLLM_QWEN4EXP_QSA_STATIC_BUCKET", "auto").strip().lower()
    if pref in ("0", "false", "no", "off"):
        return False
    if pref in ("1", "true", "yes", "on"):
        return True
    return device_type == "hpu" and is_hpu_lazy_mode()


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``Qwen4ExpTextRMSNorm``: float norm, multiply by ``1 + w``, cast back.

    H23-4 measured the fused HPU op here and **rejected** it: this function is
    the indexer's norm (query rows in ``QSAIndexer.project_qk`` and the pooled
    block rows in ``pool_indexer_blocks``), and a 1-ULP change there reorders
    tied block scores. With the fused op the sparse chunk of
    ``test_qsa_layer_hpu.py`` went from 8.79e-03 to **4.21e-02** HPU-vs-CPU
    absmax (one swapped block), i.e. through the 2e-2 gate. The *main* q/k
    attention norms have no such discrete consumer and do use the fused op
    (``qsa.Qwen4ExpQSAAttention._project_qkv_gate``); see :func:`hpu_rms_norm`.
    """
    # Materialize the BF16 input (lazy only; plain contiguous in eager):
    # as a segment-internal tensor (e.g. a GEMM output view) the lazy
    # compiler eliminates the BF16 rounding of the producer and feeds
    # unrounded fp32 values into the norm.
    x = _materialized(x)
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * (1.0 + weight.float())
    return out.to(dtype=x.dtype)


def hpu_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """``torch.ops.hpu.rms_norm`` in Gemma form, or None to take the torch path.

    H23-4: the eager Gemma RMSNorm is 7-8 launches (``float``, ``pow``,
    ``mean``, ``rsqrt``, ``mul``, ``add``, ``mul``, ``to``) and measured
    127-147 us per call at the QSA shapes; the fused HPU op is 33-39 us
    (3.3-4x). Its ``addOneToWeight=True`` flag is exactly
    ``x * rsqrt(mean(x^2)+eps) * (1+w)``, i.e. the same formula, not an
    approximation. Kept behind ``VLLM_QWEN4EXP_QSA_H23_FUSE`` with a strict
    shape/dtype/device guard; anything unusual takes the torch path.

    Only the *attention* q/k norms may use it: the indexer and pooled-block
    norms keep the exact torch chain, because a 1-ULP change there reorders
    tied block scores and fails the layer gate (see :func:`gemma_rmsnorm`).
    """
    if not _h23_enabled("fuse"):
        return None
    if x.device.type != "hpu" or x.dtype != torch.bfloat16:
        return None
    if weight.dtype != torch.bfloat16 or weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        return None
    weight = weight.detach()
    if weight.device != x.device:
        return None
    try:
        op = torch.ops.hpu.rms_norm
    except AttributeError:  # pragma: no cover - older Habana runtime
        return None
    if _is_hpu_lazy_mode():
        x = _materialized(x)
    return op(x, weight, float(eps), None, True)[0]


def apply_partial_rope(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE on ``x[..., :rotary_dim]``.

    ``cos_sin`` is ``[..., rotary_dim]`` with cos and sin concatenated, the
    layout of vLLM's ``cos_sin_cache`` rows. Broadcasts over a head axis.
    """
    rotary_dim = cos_sin.shape[-1]
    cos, sin = cos_sin.chunk(2, dim=-1)
    # Materialize the rotated input first (lazy only; plain contiguous in
    # eager): the producer (usually a GemmaRMSNorm ending in a BF16 cast)
    # must execute as its own segment and hand over real BF16 storage, or
    # the compiler fuses it into the RoPE products below and its output
    # flips by 1 ULP on ~20% of rows (see the product note below).
    rotated = _materialized(x[..., :rotary_dim])
    while cos.ndim < rotated.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x1, x2 = rotated.chunk(2, dim=-1)
    # Materialize each product before the add/sub (lazy mode only; a plain
    # contiguous tensor in eager, so eager values are unchanged). Fused into
    # one lazy graph segment, the compiler evaluates x1*cos - x2*sin without
    # the per-product BF16 roundings the eager/CPU path applies, and ~20% of
    # the rotated rows flip by 1 ULP -- enough to flip tied indexer blocks in
    # the selection downstream. Real storage per product restores the exact
    # rounding boundaries.
    rope = torch.cat(
        (
            _materialized(x1 * cos) - _materialized(x2 * sin),
            _materialized(x2 * cos) + _materialized(x1 * sin),
        ),
        dim=-1,
    )
    if rotary_dim == x.shape[-1]:
        return rope
    return torch.cat((rope, x[..., rotary_dim:]), dim=-1)


def _apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Match ``vllm...mrope.apply_interleaved_rope`` (chunked T/H/W -> THW)."""
    channels = torch.arange(x.shape[-1], device=x.device)
    is_height = (channels % 3 == 1) & (channels < mrope_section[1] * 3)
    is_width = (channels % 3 == 2) & (channels < mrope_section[2] * 3)
    result = torch.where(is_height, x[1], x[0])
    return torch.where(is_width, x[2], result)


def mrope_cos_sin(
    cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    mrope_section: list[int] | None = None,
    mrope_interleaved: bool = False,
    equal_rows: bool | None = None,
) -> torch.Tensor:
    """``[N, rotary_dim]`` cos||sin rows for ``[N]`` or ``[3, N]`` positions.

    Matches ``MRotaryEmbedding.forward_native`` cos/sin composition. ``cache``
    is ``[max_pos, rotary_dim]``.

    ``equal_rows`` is the H28 text-only fast path: when the three M-RoPE rows
    are the same positions, the interleave below only ever selects values from
    a row that equals row 0, so the gather collapses to ``positions[0]`` and
    the composed rows are the raw cache rows for it -- bit-identical, and one
    launch instead of the 3N-row gather plus the four ``torch.where`` ops (see
    :func:`_apply_interleaved_rope`). ``None`` asks the bound host control
    (which already holds a CPU copy of this step's positions); unknown stays
    ``None`` and takes the general path, so no device→host read is ever added.
    """
    if positions.ndim == 1:
        return cache.index_select(0, positions.to(dtype=torch.long).clamp_min(0))
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError(
            f"QSA RoPE positions must be [N] or [3, N], got {tuple(positions.shape)}"
        )
    if not mrope_section:
        raise ValueError("MRoPE positions require mrope_section")
    if equal_rows is None:
        equal_rows = _positions_rows_equal(positions)
    if equal_rows:
        return cache.index_select(
            0, positions[0].to(dtype=torch.long).clamp_min(0))
    # Runner bucket padding marks pad rows with -1: index_select raises and
    # advanced indexing would silently read the last cache row. Pad rows are
    # discarded downstream, so clamp to a valid position.
    gathered = cache[positions.to(dtype=torch.long).clamp_min(0)]
    cos, sin = gathered.chunk(2, dim=-1)
    if mrope_interleaved:
        cos = _apply_interleaved_rope(cos, mrope_section)
        sin = _apply_interleaved_rope(sin, mrope_section)
    else:
        cos = torch.cat(
            [part[i] for i, part in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [part[i] for i, part in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1,
        )
    return torch.cat((cos, sin), dim=-1)


def select_qsa_indices(
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    eps: float,
    block_cos_sin: torch.Tensor | None,
    token_budget: int,
    compress_ratio: int,
    host_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[T, budget + ratio - 1]`` int32 token indices, ``-1`` padded.

    ``q`` is ``[T, heads, D]`` already normed and rotated. ``raw_keys`` is
    ``[S, D]`` unnormalized and unrotated, stored at logical positions
    ``0 .. S-1``. ``block_cos_sin`` is ``[num_complete_blocks, rotary_dim]``
    for block starts ``0, ratio, 2*ratio, ...`` (ignored when no block fits).
    """
    if token_budget % compress_ratio != 0:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    if q.ndim != 3:
        raise ValueError(f"q must be [T, heads, D], got {tuple(q.shape)}")
    num_queries, _, head_dim = q.shape
    num_keys = raw_keys.shape[0]
    if query_positions.shape != (num_queries,):
        raise ValueError("query_positions must be [T]")
    check_position_bound(host_positions, query_positions, num_keys)

    ratio = compress_ratio
    width = token_budget + ratio - 1
    block_topk = token_budget // ratio
    out = torch.full(
        (num_queries, width),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    if num_queries == 0:
        return out

    num_blocks = num_keys // ratio
    positions = query_positions.to(dtype=torch.long)
    visible = positions + 1
    n_complete = torch.div(visible, ratio, rounding_mode="floor")
    tail_len = visible - n_complete * ratio

    if num_blocks > 0:
        if block_cos_sin is None or block_cos_sin.shape[0] < num_blocks:
            raise ValueError("block_cos_sin must cover every complete key block")
        groups = raw_keys[: num_blocks * ratio].view(num_blocks, ratio, head_dim)
        pooled = pool_indexer_blocks(
            groups,
            k_norm_weight=k_norm_weight,
            eps=eps,
            block_cos_sin=block_cos_sin[:num_blocks],
        )
        n_block_tokens = _packed_block_topk(
            q,
            pooled,
            n_complete,
            num_blocks,
            min(block_topk, num_blocks),
            ratio,
            out,
        )
    else:
        n_block_tokens = torch.zeros(num_queries, dtype=torch.long, device=q.device)

    _append_tail(out, n_complete, tail_len, n_block_tokens, ratio)
    return out


def check_position_bound(
    host_positions: torch.Tensor | None,
    query_positions: torch.Tensor,
    limit: int,
) -> None:
    """Raise if any query position is outside the key cache.

    H23-2: this is a debug assertion, and on device it is a *host sync*:
    ``int(x.max())`` measured 92-117 us standalone in ``probe_h23_ops.py`` and
    a read that has to drain the layer's enqueued GEMMs measured +120 us, per
    QSA layer, i.e. 12 drains per decode token. The selection is called on
    ``query_positions`` that ``_commit`` built as ``arange(start, start + n)``
    over the cache's own index space, so the bound holds by construction;
    the assertion therefore only runs when it is free (a host twin is bound)
    or when it is explicitly asked for (``VLLM_QWEN4EXP_QSA_DEBUG=1`` or the
    caller passing ``validate=True``). ``VLLM_QWEN4EXP_QSA_H23_SPANS=0``
    restores the unconditional device read.
    """
    bound = host_positions if host_positions is not None else query_positions
    if bound.numel() == 0:
        return
    if host_positions is None and _h23_enabled("spans") and not _validate_positions_env():
        return
    if int(bound.reshape(-1).max()) >= limit:
        raise ValueError("query position is outside the raw-key cache")


def _validate_positions_env() -> bool:
    return os.environ.get("VLLM_QWEN4EXP_QSA_DEBUG", "") == "1"


def _append_tail(
    out: torch.Tensor,
    n_complete: torch.Tensor,
    tail_len: torch.Tensor,
    n_block_tokens: torch.Tensor | int,
    ratio: int,
) -> None:
    """Append the incomplete tail tokens after the packed block tokens.

    Static-shape ``scatter_`` spelling of the previous boolean-mask
    assignment. Tail columns are strictly increasing per row, invalid tail
    slots write ``-1`` over the ``-1`` padding already in ``out``, and no
    tail column overlaps a packed block column, so the result is identical
    while avoiding a data-dependent (nonzero-style) index on HPU.

    ``n_block_tokens`` may be a plain int (the all-blocks-visible fast path in
    :func:`_packed_block_topk`, where it is ``k * ratio`` by construction);
    it then broadcasts instead of contributing a tensor op.
    """
    if ratio <= 1:
        return
    tail_off = torch.arange(ratio - 1, device=out.device)
    tail_pos = n_complete.unsqueeze(-1) * ratio + tail_off
    tail_ok = tail_off.unsqueeze(0) < tail_len.unsqueeze(1)
    if isinstance(n_block_tokens, int):
        tail_cols = (n_block_tokens + tail_off).unsqueeze(0)
    else:
        tail_cols = n_block_tokens.unsqueeze(-1) + tail_off
    src = torch.where(tail_ok, tail_pos, torch.full_like(tail_pos, -1))
    out.scatter_(1, tail_cols, src.to(dtype=torch.int32))


def pool_indexer_blocks(
    block_keys: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    eps: float,
    block_cos_sin: torch.Tensor,
) -> torch.Tensor:
    """Pool complete blocks ``[num_blocks, ratio, D]`` -> normed/rotated ``[num_blocks, D]``.

    Float-mean over the block, Gemma-RMSNorm with the indexer ``k_layernorm``
    weight, then partial RoPE at the block start. Purely incremental: callers
    pool only newly completed blocks, so the cost is amortized over the
    prefill and the pooled cache never recompute old rows.
    """
    # Materialize the BF16 block mean (lazy only; plain contiguous in
    # eager): it is a rounding boundary, but as a segment-internal tensor
    # the lazy compiler eliminates the float->BF16->float roundtrip and
    # the norm below then reads the unrounded fp32 mean, flipping the
    # pooled rows by 1 ULP on ~20% of entries.
    pooled = gemma_rmsnorm(
        _materialized(block_keys.float().mean(dim=1).to(dtype=block_keys.dtype)),
        k_norm_weight,
        eps,
    )
    return apply_partial_rope(pooled, block_cos_sin)


def _topk_select_applies(scores: torch.Tensor, k: int) -> bool:
    # Off by default: measured on the card (tools/qwen4exp_flashnext/kernel_ab/topk_probe2.py, 512 rows), the HPU top-k
    # with k=512 is slower than the full stable sort it replaces (65,536 columns: 30.9 vs 16.9 ms;
    # 2,048: 0.81 vs 0.34 ms). Bit-identical either way.
    if os.environ.get("VLLM_QWEN4EXP_QSA_TOPK_SELECT", "0") == "0":
        return False
    return 0 < k < scores.shape[-1] and scores.shape[-1] >= int(
        os.environ.get("VLLM_QWEN4EXP_QSA_TOPK_MIN_BLOCKS", "2048"))


def stable_topk_desc(scores: torch.Tensor, k: int) -> torch.Tensor:
    """``argsort(scores, dim=-1, stable=True, descending=True)[:, :k]`` without sorting every column.

    H31: the prefill selection sorted all pooled blocks of every query to keep 512 of them; at a
    200K context that is 2048 x 50,000 elements per QSA layer per chunk. Here one ``topk`` finds the
    k-th largest score, the ties at that score are filled in index order (what a stable sort does),
    a second ``topk`` over distinct keys lists the chosen columns in ascending index order, and a
    stable sort of those ``k`` scores gives the final order. Same columns, same order, so the
    selection is bit-identical; ``-inf`` (invisible) blocks are ordinary ties of the lowest score.
    """
    kth = torch.topk(scores, k, dim=-1, sorted=False).values.min(dim=-1, keepdim=True).values
    above = scores > kth
    tied = scores == kth
    need = k - above.sum(dim=-1, keepdim=True)
    take = above | (tied & (torch.cumsum(tied.to(torch.int32), dim=-1) <= need))
    columns = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.float32)
    key = torch.where(take, -columns, torch.full_like(columns, float("-inf")))
    chosen = torch.topk(key, k, dim=-1, sorted=True).indices
    order = torch.argsort(scores.gather(1, chosen), dim=-1, stable=True, descending=True)
    # The HPU cannot compile a gather of int64 values; the column ids fit int32.
    return chosen.to(torch.int32).gather(1, order).to(torch.int64)


def _packed_block_topk(
    q: torch.Tensor,
    pooled: torch.Tensor,
    n_complete: torch.Tensor,
    block_cap: int,
    k: int,
    ratio: int,
    out: torch.Tensor,
    all_blocks_visible: bool = False,
) -> torch.Tensor | int:
    """Score ``q`` against ``pooled`` blocks, pack the top-``k`` into ``out``.

    ``pooled`` may be wider than the valid range (compile-stable window):
    blocks past ``n_complete`` (per query) or ``block_cap`` (global) score
    ``-inf`` and never survive ``picked_ok``, so a padded window selects
    exactly what the full-history selection would. Returns per-query packed
    token counts ``[T]``.

    ``all_blocks_visible`` is the caller's host-side proof that every row of
    ``pooled`` is visible to every query (``pooled.shape[0] == block_cap`` and
    even the *first* query's ``n_complete`` reaches ``block_cap``): the
    ``-inf`` mask, the ``picked_ok`` gather and the count are then no-ops, so
    the selection returns the constant ``k * ratio`` and skips three launches
    (H23-3).
    """
    num_queries, _, head_dim = q.shape
    # Materialize the fp32 score planes (lazy only; plain contiguous in
    # eager). Segment-internal, the compiler eliminates the .float()
    # casts and runs the score product in BF16; the resulting 1-ULP score
    # flips reorder the many exact (relu-zero) ties and change the picked
    # blocks. Real fp32 storage keeps the scores at eager accuracy.
    dots = _materialized(torch.matmul(q.float(), pooled.float().transpose(0, 1)))
    scores = _materialized(torch.relu(dots).sum(dim=1) / math.sqrt(head_dim))
    if all_blocks_visible:
        valid_block = None
    else:
        block_ids = torch.arange(pooled.shape[0], device=q.device)
        if block_cap >= pooled.shape[0]:
            valid_block = block_ids.unsqueeze(0) < n_complete.unsqueeze(1)
        else:
            below_cap = block_ids < block_cap
            valid_block = (block_ids.unsqueeze(0) < n_complete.unsqueeze(1)) & below_cap.unsqueeze(0)
        scores = scores.masked_fill(~valid_block, float("-inf"))
    if k == 0:
        return torch.zeros(num_queries, dtype=torch.long, device=q.device)
    # Deterministic, width-independent tie-break. relu zeroes whole blocks of
    # negative dots, so exact score ties are common in the real model, and
    # ``torch.topk`` orders ties implementation-dependently — a 2048-wide
    # bundled window could pick a different tied subset than the full history.
    # A stable argsort of the negated scores is exactly (score desc, index
    # asc) on every width, so full and windowed selections stay bit-identical.
    #
    # H23-1/H23-3: PyTorch's *stable descending* sort breaks ties by
    # increasing index, i.e. it is the same total order as sorting the negated
    # scores, with one launch less (probe_h23_ops.py: identical indices over 30
    # tie-heavy random trials, 46.7 -> 39.5 us at width 640).
    if _h23_enabled("tile") and _topk_select_applies(scores, k):
        picked = stable_topk_desc(scores, k)
    else:
        if _h23_enabled("tile"):
            order = torch.argsort(scores, dim=-1, stable=True, descending=True)
        else:
            order = torch.argsort(-scores, dim=-1, stable=True)
        picked = order[:, :k]
    if all_blocks_visible:
        offsets = torch.arange(ratio, device=q.device)
        block_tokens = picked.unsqueeze(-1) * ratio + offsets
        flat = block_tokens.reshape(num_queries, k * ratio)
        out[:, : k * ratio] = flat.to(dtype=torch.int32)
        return k * ratio
    picked_ok = valid_block.gather(1, picked)
    offsets = torch.arange(ratio, device=q.device)
    block_tokens = picked.unsqueeze(-1) * ratio + offsets
    block_tokens = block_tokens.masked_fill(~picked_ok.unsqueeze(-1), -1)
    flat = block_tokens.reshape(num_queries, k * ratio)
    # No compaction pass is needed: invalid blocks score -inf, so the stable
    # descending argsort places every valid block before every invalid one,
    # and the -1 masking above therefore leaves ``flat`` already packed
    # (valid token groups first, -1 after). Placing it into ``out`` is a
    # plain slice copy. This replaces boolean-mask assignments
    # (``packed[rows[valid], rank[valid]] = ...`` and the masked copy into
    # ``out``), each of which costs ~3 ms on eager HPU as a data-dependent
    # index, with exactly the same resulting tensor.
    out[:, : k * ratio] = flat.to(dtype=torch.int32)
    return picked_ok.sum(dim=-1) * ratio


def select_qsa_indices_from_pooled(
    q: torch.Tensor,
    pooled: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    total_blocks: int,
    token_budget: int,
    compress_ratio: int,
    total_tokens: int | None = None,
    host_positions: torch.Tensor | None = None,
    host_min_position: int | None = None,
) -> torch.Tensor:
    """Windowed selection over a fixed-width pooled-block buffer.

    The Qwen3.8 graph-size compromise applied to the QSA indexer: ``pooled``
    is the first ``W`` rows of the pooled-block cache where ``W`` is a
    compile-stable bundle width (a multiple of the context tile). Rows past
    ``total_blocks`` are zero padding and score ``-inf``. ``k`` stays the
    fixed ``token_budget // ratio`` so every tensor shape — and therefore
    every compiled program — depends only on the bundle, never on the real
    context length. Bit-identical to ``select_qsa_indices`` on the same
    history (see test_qsa_tile_replay.py).

    An unpadded pooled cache (``pooled.shape[0] == total_blocks``, the
    incrementally maintained decode cache) is also accepted: then
    ``k = min(block_topk, total_blocks)``, exactly the cap the full-history
    path applies, so the selection is unchanged and no zero padding or
    re-pooling of old blocks is needed.

    ``host_min_position`` is the smallest query position of this chunk *as a
    Python int*, when the caller knows it without a device read (``qsa.py``
    does: the committed chunk always occupies cache rows
    ``S_kv - n .. S_kv - 1``). If even that first query already sees every
    pooled block, the ``-inf`` mask, the gather and the count are provably
    no-ops and are skipped (H23-3).
    """
    if token_budget % compress_ratio != 0:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    if q.ndim != 3:
        raise ValueError(f"q must be [T, heads, D], got {tuple(q.shape)}")
    num_queries, _, _ = q.shape
    if pooled.ndim != 2:
        raise ValueError(f"pooled must be [W, D], got {tuple(pooled.shape)}")
    if query_positions.shape != (num_queries,):
        raise ValueError("query_positions must be [T]")
    if total_blocks > pooled.shape[0]:
        raise ValueError("pooled window is smaller than the real block count")
    # Materialize the score inputs (lazy only; plain contiguous in eager).
    # In the layer forward these arrive straight from the cache append /
    # indexer projection in the same lazy segment, and the compiler fuses
    # that chain into the score GEMM (dropping the BF16 boundaries on the
    # way); with real storage the scores compute as they do in isolation,
    # at eager accuracy, and far fewer tied blocks flip.
    q = _materialized(q)
    pooled = _materialized(pooled)
    # The incomplete tail block may hold up to ``ratio - 1`` tokens past the
    # completed-block range; ``total_tokens`` pins the exact bound when known.
    token_bound = (total_tokens if total_tokens is not None
                   else total_blocks * compress_ratio + compress_ratio)
    check_position_bound(host_positions, query_positions, token_bound)

    ratio = compress_ratio
    width = token_budget + ratio - 1
    block_topk = token_budget // ratio
    out = torch.full(
        (num_queries, width),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    if num_queries == 0:
        return out

    positions = query_positions.to(dtype=torch.long)
    visible = positions + 1
    n_complete = torch.div(visible, ratio, rounding_mode="floor")
    tail_len = visible - n_complete * ratio
    k = min(block_topk, pooled.shape[0])
    all_visible = (
        host_min_position is not None
        and pooled.shape[0] == total_blocks
        and (int(host_min_position) + 1) // ratio >= total_blocks
    )
    n_block_tokens = _packed_block_topk(
        q, pooled, n_complete, total_blocks, k, ratio, out,
        all_blocks_visible=all_visible,
    )

    _append_tail(out, n_complete, tail_len, n_block_tokens, ratio)
    return out


def _sparse_width_tile(num_queries: int, num_kv_heads: int, head_dim: int) -> int:
    """Selected-key columns per gather so the fp32 plane stays near 2 GiB.

    The one-shot plane ``[T, width, Kv, D]`` fp32 is 8.2 GiB at the warmup
    bucket and failed beside the resident experts. Splitting the key axis
    keeps every query in the GEMM (decode is not split: it never reaches
    this helper).
    """
    bytes_per_col = num_queries * num_kv_heads * head_dim * 4
    if bytes_per_col <= 0:
        return 1
    return max(1, min(512, (2 << 30) // bytes_per_col))


def qsa_attention_path(
    num_queries: int,
    width: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    device_type: str | None = None,
    exact_pref: str | None = None,
    dense: bool = False,
) -> str:
    """``"exact"`` (fp32 products) or ``"fast"`` (BF16 GEMM) for one call.

    Split out of :func:`sparse_attention` so the routing rule — the thing that
    decides whether a step runs the exact fp32 path or the fast BF16 one — is
    a pure function a CPU test can exercise. The rule:

    * non-HPU tensors (the offline reference path) always run exact;
    * ``VLLM_QWEN4EXP_QSA_EXACT=1`` forces exact, ``=0`` forces fast;
    * the dense prefix path (H12's ``dense_prefix_attention``: every decode
      step and short prefill, ``S_kv <= indexer_budget``) runs fast on HPU
      under H23-1 — the Gaudi2 MME accumulates a BF16 GEMM in fp32 in
      hardware, so the fp32 planes only bought the ``k``/``v`` copy of the
      whole cache per token (measured 1.56x at decode T=1, S_kv=257);
    * the gathered sparse path runs fast too under H28-1. H23 kept it on the
      fp32 plane rule because the fast variant *gathered* the selected rows
      once per 1024-column tile: at decode T=1 the whole gather is 2 MiB, yet
      ``width``=2051 was split into ``1024 + 1024 + 3`` and the clamp,
      ``index_select`` and GEMM ran three times for K and three times for V
      (measured 0.45x, i.e. 2.2x slower, at S_kv=2560). With the column cap
      gone the same call is one gather and one GEMM per product;
    * the fp32 plane rule below survives only for the sparse path with the
      H23-1 switch off (the pre-H28 behavior an A/B run needs).
    """
    exact_plane = num_queries * width * num_kv_heads * head_dim * 4
    device_type = device_type if device_type is not None else "cpu"
    exact_pref = (os.environ.get("VLLM_QWEN4EXP_QSA_EXACT", "")
                  if exact_pref is None else exact_pref)
    if device_type != "hpu":
        return "exact"
    if exact_pref == "0":
        return "fast"
    if exact_pref == "1":
        return "exact"
    if dense and _h23_enabled("attn"):
        return "fast"
    if _h23_enabled("attn") and _h28_enabled("sparse"):
        return "fast"
    return "fast" if exact_plane > _EXACT_PLANE_BUDGET else "exact"


def effective_qsa_width(width: int, total_tokens: int) -> int:
    """Gather width actually needed for a packed QSA selection.

    Selection packs every valid token first (see ``_packed_block_topk`` and
    ``_append_tail``), and a query can select at most its visible causal
    tokens, so no row ever carries a valid entry past column
    ``total_tokens``. Clamping the fixed ``budget + ratio - 1`` width to
    ``min(width, total_tokens)`` therefore drops only ``-1`` pads.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    return min(int(width), int(total_tokens)) if total_tokens else int(width)


def clamp_qsa_indices(indices: torch.Tensor, total_tokens: int) -> torch.Tensor:
    """Slice a packed selection to :func:`effective_qsa_width` (no copy if full)."""
    width = effective_qsa_width(indices.shape[-1], total_tokens)
    return indices if width == indices.shape[-1] else indices[:, :width]


def dense_prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    query_positions: torch.Tensor | None = None,
    valid_num_keys: int | None = None,
) -> torch.Tensor:
    """Attention over the whole active KV prefix, with no index gather.

    This is the short-context / decode fast path: when the total KV length
    fits the indexer budget every causal token is selected, so gathering the
    selected rows would only re-read ``k``/``v`` (plus thousands of dummy
    row-0 pads in decode). ``k``/``v`` here are already the active prefix
    ``[:, :S_kv]`` views of the cache, and the math is the same fp32-softmax
    attention as :func:`sparse_attention` (BF16 GEMMs on the "fast" HPU
    route, fp32 products on the "exact" route; see :func:`qsa_attention_path`).

    ``query_positions=None`` means every query attends every key. That is
    exactly the decode case (``q_len == 1``): the single query sits at
    ``S_kv - 1``, after every cached key, so no causal mask is built or
    applied at all. With positions given (short prefill), keys after each
    query position are masked, matching the causal selection.

    Static-bucket (lazy) form: pass the *backing capacity* tensors as
    ``k``/``v`` (e.g. 512 rows) together with ``valid_num_keys=S_kv``.
    Positions ``>= S_kv`` are masked with ``-inf`` (``finfo.min`` before
    softmax, exactly like the causal mask) and contribute exactly zero
    after softmax, so the result is the active-prefix result while the
    GEMM/softmax shapes stay at the capacity until the buffer grows --
    no per-token recipe recompilation in lazy mode. Callers must ensure
    rows past ``valid_num_keys`` are finite (the QSA slot zero-fills new
    backing capacity, so its padding rows are zero).
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [tokens, heads, dim]")
    num_queries, num_q_heads, head_dim = q.shape
    num_keys = k.shape[0]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    if num_queries == 0:
        return q.new_empty(0, num_q_heads, head_dim)
    if num_keys == 0:
        return q.new_zeros(num_queries, num_q_heads, head_dim)
    if valid_num_keys is None:
        valid_num_keys = num_keys
    if not 0 <= valid_num_keys <= num_keys:
        raise ValueError(
            f"valid_num_keys must be in [0, {num_keys}], got {valid_num_keys}")
    if valid_num_keys == 0:
        return q.new_zeros(num_queries, num_q_heads, head_dim)
    if query_positions is not None and query_positions.shape != (num_queries,):
        raise ValueError("query_positions must be [T]")

    n_rep = num_q_heads // num_kv_heads
    path = qsa_attention_path(
        num_queries,
        num_keys,
        num_kv_heads,
        head_dim,
        device_type=q.device.type,
        dense=True,
    )
    if (path == "fast" and query_positions is not None and valid_num_keys == num_keys
            and os.environ.get("VLLM_QWEN4EXP_QSA_DENSE_KVMAJOR", "1") != "0"):
        return _dense_prefix_attention_kv_major(q, k, v, scale, query_positions, n_rep)
    # One batched GEMM per product, batch = KV head (no per-head loop):
    # q as [Kv, T * rep, D] against k^T [Kv, D, S] / v [Kv, S, D].
    q_b = q.view(num_queries, num_kv_heads, n_rep, head_dim)
    q_b = q_b.permute(1, 0, 2, 3).reshape(num_kv_heads, num_queries * n_rep, head_dim)
    if path == "fast":
        scores_b = torch.matmul(q_b, k.permute(1, 2, 0)) * scale
    else:
        scores_b = torch.matmul(q_b.float(), k.permute(1, 2, 0).float()) * scale
    scores = scores_b.reshape(num_kv_heads, num_queries, n_rep, num_keys)
    scores = scores.permute(1, 0, 2, 3).float()
    del scores_b

    row_has = None
    # Static-bucket mask: keys at/past valid_num_keys are capacity padding.
    # Without query_positions (decode) this is the only mask; with them it
    # is subsumed by the causal mask (positions < valid_num_keys) but is
    # applied anyway so the two conditions stay explicit and independent.
    static_invalid = None
    if valid_num_keys < num_keys:
        static_invalid = (torch.arange(num_keys, device=q.device)
                          >= valid_num_keys).view(1, 1, 1, num_keys)
    if query_positions is not None:
        positions = query_positions.to(dtype=torch.long)
        invalid = torch.arange(num_keys, device=q.device).unsqueeze(0) > positions.unsqueeze(1)
        scores = scores.masked_fill(
            invalid[:, None, None, :], torch.finfo(torch.float32).min)
        row_has = ~invalid.all(dim=-1)
        scores = torch.where(
            row_has[:, None, None, None], scores, torch.zeros_like(scores))
    if static_invalid is not None:
        scores = scores.masked_fill(static_invalid, torch.finfo(torch.float32).min)
        if row_has is None:
            # Every query has valid_num_keys (>= 1 here) valid keys.
            row_has = torch.ones(
                num_queries, device=q.device, dtype=torch.bool)
    attn = torch.softmax(scores, dim=-1)
    del scores
    if row_has is not None:
        attn = attn.masked_fill(~row_has[:, None, None, None], 0)

    attn_b = attn.permute(1, 0, 2, 3).reshape(num_kv_heads, num_queries * n_rep, num_keys)
    del attn
    if path == "fast":
        out_b = torch.matmul(attn_b.to(dtype=v.dtype), v.permute(1, 0, 2))
    else:
        out_b = torch.matmul(attn_b, v.permute(1, 0, 2).float())
    out = out_b.reshape(num_kv_heads, num_queries, n_rep, head_dim)
    return out.permute(1, 0, 2, 3).reshape(num_queries, num_q_heads, head_dim).to(dtype=q.dtype)


def _dense_prefix_attention_kv_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    query_positions: torch.Tensor,
    n_rep: int,
) -> torch.Tensor:
    """The "fast" causal branch of :func:`dense_prefix_attention` without its permutes.

    That branch moves the fp32 score plane ``[T, Kv, rep, S]`` to query-major order and back, and
    at a 2048-token prefill each pass copies 400 MB per layer. Here the plane stays in the GEMM's
    ``[Kv, T * rep, S]`` order: the mask, the softmax and the value GEMM are row-wise, so every
    element goes through the same operations and the result is bit-identical. The query positions
    are software-cache indices (``_commit`` returns an arange starting at 0 or at the cached length),
    so every row sees key 0 and the general path's empty-row guards are no-ops; they are dropped.
    """
    num_queries, num_q_heads, head_dim = q.shape
    num_keys, num_kv_heads = k.shape[0], k.shape[1]
    q_b = q.view(num_queries, num_kv_heads, n_rep, head_dim)
    q_b = q_b.permute(1, 0, 2, 3).reshape(num_kv_heads, num_queries * n_rep, head_dim)
    scores = (torch.matmul(q_b, k.permute(1, 2, 0)) * scale).float()
    positions = query_positions.to(dtype=torch.long)
    invalid = torch.arange(num_keys, device=q.device).unsqueeze(0) > positions.unsqueeze(1)
    scores = scores.view(num_kv_heads, num_queries, n_rep, num_keys).masked_fill(
        invalid[None, :, None, :], torch.finfo(torch.float32).min)
    attn = torch.softmax(scores, dim=-1).view(num_kv_heads, num_queries * n_rep, num_keys)
    out_b = torch.matmul(attn.to(dtype=v.dtype), v.permute(1, 0, 2))
    out = out_b.reshape(num_kv_heads, num_queries, n_rep, head_dim)
    return out.permute(1, 0, 2, 3).reshape(num_queries, num_q_heads, head_dim).to(dtype=q.dtype)


class InflightGate:
    """Keep at most ``limit`` tiles of eager work queued on the card (H31).

    The host queues eager ops far ahead of the card, and a queued op keeps its temporaries
    allocated until it has run. A loop over tiles therefore held every tile's planes at once: a
    QSA prefill chunk over a 16K cache peaked 14 GiB above its start, linear in the cache length.
    ``wait()`` before a tile blocks until the tile ``limit`` places back has run; ``mark()`` after
    it records an event. With ``limit`` 2 the card always has the previous tile to work on while
    the host queues the next. 0 disables it; off HPU it does nothing.
    """

    def __init__(self, device: torch.device, limit: int | None = None) -> None:
        if limit is None:
            limit = int(os.environ.get("VLLM_QWEN4EXP_PREFILL_INFLIGHT", "2") or 0)
        self.limit = limit if device.type == "hpu" else 0
        self.events: list = []

    def wait(self) -> None:
        if self.limit and len(self.events) >= self.limit:
            self.events.pop(0).synchronize()

    def mark(self) -> None:
        if self.limit:
            event = torch.hpu.Event()
            event.record()
            self.events.append(event)


def masked_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    max_plane_bytes: int | None = None,
) -> torch.Tensor:
    """:func:`sparse_attention` over the whole cache with the unselected keys masked out.

    A prefill chunk above the token budget gathers ``T x (budget + ratio - 1)`` key and value rows
    for its queries, about 8.6 GB of traffic per layer at T=2048. While the cache is only a few
    thousand keys long, one score GEMM over all of them moves far less, and masking the keys a
    query did not select leaves exactly the selected set in its softmax. Same scores and the same
    key set per query; only the fp32 summation order differs, so the result is not bit-identical.
    Query rows are tiled so the fp32 score plane stays under ``max_plane_bytes``.
    """
    num_queries, num_q_heads, head_dim = q.shape
    num_keys, num_kv_heads = k.shape[0], k.shape[1]
    n_rep = num_q_heads // num_kv_heads
    if max_plane_bytes is None:
        max_plane_bytes = int(os.environ.get("VLLM_QWEN4EXP_QSA_MASKED_PLANE_MIB", "1024")) << 20
    rows = max(1, min(num_queries, max_plane_bytes // max(1, num_kv_heads * n_rep * num_keys * 4)))
    k_t = k.permute(1, 2, 0)
    v_t = v.permute(1, 0, 2)
    out = q.new_empty(num_queries, num_q_heads, head_dim)
    gate = InflightGate(q.device)
    for begin in range(0, num_queries, rows):
        end = min(begin + rows, num_queries)
        gate.wait()
        part = indices[begin:end]
        # HPU has no bool scatter; mark int8 cells and compare. Pads land in the extra last column.
        marks = torch.zeros(end - begin, num_keys + 1, dtype=torch.int8, device=q.device)
        slot = torch.where(part >= 0, part, torch.full_like(part, num_keys)).to(torch.int64)
        marks.scatter_(1, slot, torch.ones(slot.shape, dtype=torch.int8, device=q.device))
        selected = marks[:, :num_keys] != 0
        q_b = q[begin:end].view(end - begin, num_kv_heads, n_rep, head_dim)
        q_b = q_b.permute(1, 0, 2, 3).reshape(num_kv_heads, (end - begin) * n_rep, head_dim)
        scores = (torch.matmul(q_b, k_t) * scale).float()
        # In place: the same values as masked_fill, one fp32 plane fewer (H31).
        scores = scores.view(num_kv_heads, end - begin, n_rep, num_keys)
        scores.masked_fill_(~selected[None, :, None, :], torch.finfo(torch.float32).min)
        attn = torch.softmax(scores, dim=-1).view(num_kv_heads, (end - begin) * n_rep, num_keys)
        del scores
        out_b = torch.matmul(attn.to(dtype=v.dtype), v_t)
        out[begin:end] = out_b.view(num_kv_heads, end - begin, n_rep, head_dim).permute(1, 0, 2, 3).reshape(
            end - begin, num_q_heads, head_dim)
        gate.mark()
    return out


def fsdpa_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    max_bias_bytes: int | None = None,
) -> torch.Tensor:
    """:func:`masked_dense_attention` through the fused attention kernel (H31).

    The masked-dense form materializes fp32 score planes, masks them and runs a separate softmax:
    at a 20K-key cache that is 95 ms per layer for a 4,096-row chunk against about 6 ms of GEMM
    work. FusedSDPA scores, masks, normalizes (fp32 softmax) and multiplies by V in one kernel; the
    selection becomes an additive BF16 bias, 0 for the keys a query selected and the BF16 minimum
    elsewhere (a row with nothing selected gets a uniform softmax, as in the masked-dense form).
    GQA goes through the kernel's own five-dimensional reshape. Query rows are tiled so each bias
    plane stays under ``max_bias_bytes`` (the kernel's byte offset is 32-bit) and an in-flight gate
    keeps the tiles' biases from piling up.
    """
    from habana_frameworks.torch.hpex.kernels.FusedSDPA import gqa_input_reshape_fwd, gqa_output_reshape, is_gqa

    num_queries, num_q_heads, head_dim = q.shape
    num_keys = k.shape[0]
    if max_bias_bytes is None:
        max_bias_bytes = int(os.environ.get("VLLM_QWEN4EXP_QSA_FSDPA_BIAS_MIB", "128")) << 20
    rows = max(1, min(num_queries, max_bias_bytes // max(1, num_keys * 2)))
    q4 = q.permute(1, 0, 2).unsqueeze(0)
    k4 = k.permute(1, 0, 2).unsqueeze(0)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    out = q.new_empty(num_queries, num_q_heads, head_dim)
    neg = torch.finfo(torch.bfloat16).min
    # "fast" is the vLLM-Gaudi prompt-attention default for every model but qwen2 (the 27B serves
    # with it). Measured at T=2048, S=16K, 24 heads of 256 with a bias: fp32 21.9 ms (38 TFLOPS),
    # default 7.8 ms, fast 5.3 ms (154 TFLOPS; a plain bf16 matmul reaches 255).
    softmax_mode = os.environ.get("VLLM_QWEN4EXP_QSA_FSDPA_SOFTMAX", "fast")
    gate = InflightGate(q.device)
    for begin in range(0, num_queries, rows):
        end = min(begin + rows, num_queries)
        gate.wait()
        part = indices[begin:end]
        marks = torch.zeros(end - begin, num_keys + 1, dtype=torch.int8, device=q.device)
        slot = torch.where(part >= 0, part, torch.full_like(part, num_keys)).to(torch.int64)
        marks.scatter_(1, slot, torch.ones(slot.shape, dtype=torch.int8, device=q.device))
        bias = ((marks[:, :num_keys].to(torch.bfloat16) - 1.0) * -neg).view(1, 1, end - begin, num_keys)
        qt, kt, vt = q4[:, :, begin:end], k4, v4
        gqa = is_gqa(qt, kt)
        if gqa:
            qt, kt, vt, bias = gqa_input_reshape_fwd(qt, kt, vt, bias)
        o = torch.ops.hpu.sdpa_recomp_fwd(qt, kt, vt, bias, 0.0, scale, False, True, softmax_mode, None, "right")[0]
        if gqa:
            o = gqa_output_reshape(o)
        out[begin:end] = o[0].permute(1, 0, 2)
        gate.mark()
    return out


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """GQA attention over ``indices`` (``-1`` pad). Returns ``[T, Hq, D]``.

    Softmax always runs in fp32. On the HPU the gathered score and value
    products run as BF16 GEMMs with fp32 accumulation (the SDPA precision
    class Strata uses for its gathered QSA attention) instead of the
    fp32-materialized planes, which cost ~45 GiB of traffic per layer at the
    prefill bucket. ``VLLM_QWEN4EXP_QSA_EXACT=1`` keeps the exact fp32 path.
    A query with no selected token writes zeros instead of a NaN softmax.

    Routing is :func:`qsa_attention_path`: on HPU the BF16 GEMM path covers
    every shape this function sees (see H28-1 there); the fp32 plane path
    stays as the CPU reference and as the ``VLLM_QWEN4EXP_QSA_EXACT=1``
    oracle. Numerics inside either path are unchanged since the port was
    validated.
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [tokens, heads, dim]")
    num_queries, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Q heads must be divisible by KV heads")
    if indices.shape[0] != num_queries:
        raise ValueError("indices must have one row per query")
    if num_queries == 0:
        return q.new_empty(0, num_q_heads, head_dim)
    width = indices.shape[-1]
    exact_plane = num_queries * width * num_kv_heads * head_dim * 4
    path = qsa_attention_path(
        num_queries,
        width,
        num_kv_heads,
        head_dim,
        device_type=q.device.type,
    )
    if path == "fast":
        _qsa_debug_route(num_queries, "fast")
        return _sparse_attention_fast(q, k, v, indices, scale)
    _qsa_debug_route(num_queries, "exact")
    # One shot while the fp32 gather is under 2 GiB. That covers decode and
    # the regression tests; the warmup bucket (T=2048, width=2051) does not.
    gather_bytes = exact_plane
    if gather_bytes <= (2 << 30):
        return _sparse_attention_queries(q, k, v, indices, scale)
    return _sparse_attention_width_tiled(q, k, v, indices, scale)


def _qsa_debug_route(num_queries: int, path: str) -> None:
    """VLLM_QWEN4EXP_QSA_DEBUG=1: log the (T, path) pair once per change."""
    if os.environ.get("VLLM_QWEN4EXP_QSA_DEBUG", "") != "1":
        return
    route = (num_queries, path)
    if route != getattr(_qsa_debug_route, "_last", None):
        print(f"[qsa-route] T={num_queries} path={path}", flush=True)
        _qsa_debug_route._last = route


# fp32 gathered-plane bytes below which the exact path is essentially free
# (decode is ~17 MiB, a 60-token prompt ~250 MiB, a 2048-token chunk 8 GiB).
_EXACT_PLANE_BUDGET = 256 << 20

# Bytes of the BF16 gathered plane one fast-path tile may materialize. The
# pre-H28 helper also capped the *width* at 1024 columns, which is the same
# budget at the 512-token prefill shape (512 x 2 x 256 x 2 B = 512 KiB per
# column -> 2048 columns) but 512x too small at decode: a T=1 gather is 1 KiB
# per column, so the 2 MiB whole-tensor gather was split into 1024 + 1024 + 3.
_FAST_TILE_BYTES = 1 << 30


def _fast_width_tile(num_queries: int, num_kv_heads: int, head_dim: int) -> int:
    """Key columns per BF16 gather tile under :data:`_FAST_TILE_BYTES`.

    The return value is a *budget* bound only: a caller whose ``width`` fits
    takes the single-shot path (one gather + one GEMM per product). That
    covers every decode shape -- ``T <= 256`` yields a tile ``>= 4096``
    columns against the fixed QSA width of ``budget + ratio - 1 = 2051`` --
    while the 1024/2048-token prefill buckets keep tiles of 1024/512 columns
    and the same numerics as before.
    """
    bytes_per_col = num_queries * num_kv_heads * head_dim * 2
    if bytes_per_col <= 0:
        return 1
    return max(1, _FAST_TILE_BYTES // bytes_per_col)


def _fast_gathered_scores(
    q_mm: torch.Tensor,
    k: torch.Tensor,
    safe: torch.Tensor,
    num_queries: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """``[T, Kv, rep, W]`` BF16 score products for one gathered key block.

    ``q_mm`` is ``[T, Kv, rep, D]`` and ``safe`` the flattened, clamped
    indices of this block. The 4-D ``q`` against ``[T, Kv, D, W]`` needs no
    ``unsqueeze``/``squeeze`` pair and no per-head loop: one MME GEMM per KV
    head group, accumulated in fp32 and rounded to BF16 once, which is the
    rounding the per-tile spelling produced.
    """
    width = safe.numel() // num_queries
    k_sel = k.index_select(0, safe).view(num_queries, width, num_kv_heads, head_dim)
    return torch.matmul(q_mm, k_sel.permute(0, 2, 3, 1)) * scale


def _fast_gathered_values(
    attn: torch.Tensor,
    v: torch.Tensor,
    safe: torch.Tensor,
    num_queries: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """``[T, Kv, rep, D]`` output of one gathered value block."""
    width = safe.numel() // num_queries
    v_sel = v.index_select(0, safe).view(num_queries, width, num_kv_heads, head_dim)
    return torch.matmul(attn.to(dtype=v.dtype), v_sel.permute(0, 2, 1, 3))


def _sparse_attention_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """SDPA-class gathered attention: BF16 GEMMs, fp32 mask/softmax.

    No fp32 copy of any gathered plane. Score products and attention-weight
    products accumulate in fp32 inside the GEMM and round to BF16 once —
    the same rounding flash attention applies. Everything between the GEMMs
    (masking, softmax) stays fp32, so pad columns still contribute exactly
    zero weight.

    H28-1: when the whole selection fits one tile (every decode shape) the
    clamp, the ``to(long)`` and the gathered ``index_select`` run once for K
    and the *same* index tensor is reused for V; there is no fp32 output
    buffer and no ``out +=``. Wider widths keep the key-column tiling but
    still share one clamped index tensor across the K and V loops.
    """
    num_queries, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    n_rep = num_q_heads // num_kv_heads
    width = indices.shape[-1]
    tile = _fast_width_tile(num_queries, num_kv_heads, head_dim)
    q_g = q.view(num_queries, num_kv_heads, n_rep, head_dim)
    # Gaudi2's MME accumulates a BF16 GEMM in fp32 in hardware, so a q that
    # arrives in another dtype (an fp32 layer test) is rounded once here
    # instead of forcing the exact fp32 planes back in.
    q_mm = q_g if q_g.dtype == k.dtype else q_g.to(dtype=k.dtype)
    # One clamp + cast for the whole call; the K and V gathers share it.
    if (indices.dtype == torch.int32 and q.device.type == "hpu"
            and os.environ.get("VLLM_QWEN4EXP_QSA_INT32_GATHER", "1") != "0"):
        # H30: the HPU gather converts an int64 index back to int32; at a 2048-row prefill above the
        # budget that round trip is ~18k cast launches and ~75 ms per chunk. Same rows either way.
        safe_all = indices.clamp(min=0)
    else:
        safe_all = indices.clamp(min=0).to(dtype=torch.long)
    valid = indices >= 0
    single = tile >= width
    if single:
        safe = safe_all.reshape(-1)
        scores = _fast_gathered_scores(
            q_mm, k, safe, num_queries, num_kv_heads, head_dim, scale
        ).to(dtype=torch.float32)
    else:
        scores = q.new_empty(
            num_queries, num_kv_heads, n_rep, width, dtype=torch.float32)
        for begin in range(0, width, tile):
            end = min(begin + tile, width)
            part = safe_all[:, begin:end].reshape(-1)
            # Straight into the fp32 plane: the assignment does the BF16 ->
            # fp32 widening, as the pre-H28 in-place store did.
            scores[..., begin:end] = _fast_gathered_scores(
                q_mm, k, part, num_queries, num_kv_heads, head_dim, scale)
    scores = scores.masked_fill(
        ~valid[:, None, None, :], torch.finfo(torch.float32).min)
    row_has = valid.any(dim=-1)
    # A query with no selected token has every column at -inf; zeroing the
    # whole row (one broadcast masked_fill, no zeros_like plane) gives it a
    # uniform softmax, which the second mask then drops to exactly zero.
    scores = scores.masked_fill(~row_has[:, None, None, None], 0)
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~row_has[:, None, None, None], 0)
    del scores
    if single:
        out = _fast_gathered_values(
            attn, v, safe, num_queries, num_kv_heads, head_dim)
    else:
        out = torch.zeros(
            num_queries, num_kv_heads, n_rep, head_dim, dtype=torch.float32,
            device=q.device,
        )
        for begin in range(0, width, tile):
            end = min(begin + tile, width)
            part = safe_all[:, begin:end].reshape(-1)
            out += _fast_gathered_values(
                attn[..., begin:end], v, part, num_queries, num_kv_heads, head_dim)
    return out.reshape(num_queries, num_q_heads, head_dim).to(dtype=q.dtype)


def _sparse_attention_queries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One query tile. ``k``/``v`` stay BF16 until this tile's fp32 gather."""
    num_queries, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    n_rep = num_q_heads // num_kv_heads
    safe = indices.clamp(min=0).to(dtype=torch.long)
    flat = safe.reshape(-1)
    k_sel = k.index_select(0, flat).view(num_queries, -1, num_kv_heads, head_dim)
    q_g = q.view(num_queries, num_kv_heads, n_rep, head_dim)
    scores = torch.einsum("thrd,tkhd->thrk", q_g.float(), k_sel.float()) * scale
    del k_sel
    valid = indices >= 0
    scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(torch.float32).min)
    row_has = valid.any(dim=-1)
    scores = torch.where(row_has[:, None, None, None], scores, torch.zeros_like(scores))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~row_has[:, None, None, None], 0)
    del scores
    v_sel = v.index_select(0, flat).view(num_queries, -1, num_kv_heads, head_dim)
    out = torch.einsum("thrk,tkhd->thrd", attn, v_sel.float())
    return out.reshape(num_queries, num_q_heads, head_dim).to(dtype=q.dtype)


def _sparse_attention_width_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Same dots as ``_sparse_attention_queries``, gathered in key-width tiles."""
    num_queries, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    width = indices.shape[-1]
    n_rep = num_q_heads // num_kv_heads
    tile = _sparse_width_tile(num_queries, num_kv_heads, head_dim)
    q_g = q.view(num_queries, num_kv_heads, n_rep, head_dim).float()
    scores = q.new_empty(num_queries, num_kv_heads, n_rep, width, dtype=torch.float32)
    for begin in range(0, width, tile):
        end = min(begin + tile, width)
        part = indices[:, begin:end]
        safe = part.clamp(min=0).to(dtype=torch.long).reshape(-1)
        k_sel = k.index_select(0, safe).view(num_queries, end - begin, num_kv_heads, head_dim)
        scores[..., begin:end] = torch.einsum("thrd,tkhd->thrk", q_g, k_sel.float()) * scale
        del k_sel
    valid = indices >= 0
    scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(torch.float32).min)
    row_has = valid.any(dim=-1)
    scores = torch.where(row_has[:, None, None, None], scores, torch.zeros_like(scores))
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~row_has[:, None, None, None], 0)
    del scores
    out = torch.zeros(
        num_queries, num_kv_heads, n_rep, head_dim, dtype=torch.float32, device=q.device,
    )
    for begin in range(0, width, tile):
        end = min(begin + tile, width)
        part = indices[:, begin:end]
        safe = part.clamp(min=0).to(dtype=torch.long).reshape(-1)
        v_sel = v.index_select(0, safe).view(num_queries, end - begin, num_kv_heads, head_dim)
        out = out + torch.einsum("thrk,tkhd->thrd", attn[..., begin:end], v_sel.float())
        del v_sel
    return out.reshape(num_queries, num_q_heads, head_dim).to(dtype=q.dtype)
