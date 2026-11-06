# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA attention for the Gaudi port.

The 12 full-attention layers own the checkpoint indexer
(``index_qk_proj`` / ``q_layernorm`` / ``k_layernorm``) and run the
transformers selection rule in torch. Main K/V and the indexer raw keys live
in a per-batch-slot software cache so a prefill longer than
``indexer_budget`` and the following decode steps stay sparse-exact.

This is not the paged HPU kernel. ``self.attn`` is still constructed so the
vLLM cache manager sees a full-attention spec; the torch path does not read
that cache. Swapping the attend call for a TPC gather is the hardware step.
"""

from __future__ import annotations

import os

# Before torch / any HPU graph compile (see _gc_kernel_path). The package
# __init__ already runs this; repeating it here keeps the guarantee if
# this module is ever loaded under a different package name.
from ._gc_kernel_path import ensure_gc_kernel_path

ensure_gc_kernel_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402

from vllm.config import VllmConfig  # noqa: E402
from vllm.logger import init_logger  # noqa: E402
from vllm.model_executor.layers.layernorm import GemmaRMSNorm  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention  # noqa: E402

from ._materialize import in_compile_only_mode as _in_compile_only_mode  # noqa: E402
from ._materialize import is_hpu_lazy_mode as _is_hpu_lazy_mode  # noqa: E402
from ._materialize import lazy_mark_step as _lazy_mark_step  # noqa: E402
from ._materialize import materialized as _materialized  # noqa: E402
from .control import cached_mrope_cos_sin as _cached_mrope_cos_sin  # noqa: E402
from .control import cached_span_plan as _cached_span_plan  # noqa: E402
from .h23 import h23_enabled as _h23_enabled  # noqa: E402
from .h28 import h28_enabled as _h28_enabled  # noqa: E402
from .qsa_math import (  # noqa: E402
    apply_partial_rope,
    clamp_qsa_indices,
    dense_prefix_attention,
    gemma_rmsnorm,
    hpu_rms_norm,
    mrope_cos_sin,
    pool_indexer_blocks,
    qsa_static_bucket_enabled,
    select_qsa_indices_from_pooled,
    InflightGate,
    fsdpa_masked_attention,
    masked_dense_attention,
    sparse_attention,
)

logger = init_logger(__name__)

__all__ = ["QSAIndexer", "QSACacheState", "Qwen4ExpQSAAttention"]


def _linear_out(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Vendor linears return a bare tensor when ``return_bias=False``."""
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def _is_plain_linear(layer: nn.Module) -> bool:
    """True when ``layer`` applies an unquantized GEMM (no quant method)."""
    method = getattr(layer, "quant_method", None)
    return isinstance(method, UnquantizedLinearMethod)


def _request_spans(
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
) -> list[tuple[int, int]]:
    if query_start_loc is None:
        return [(0, num_tokens)]
    loc = query_start_loc.tolist()
    return [(int(loc[i]), int(loc[i + 1])) for i in range(len(loc) - 1)]


def _slice_positions(positions: torch.Tensor, start: int, end: int) -> torch.Tensor:
    if positions.ndim == 1:
        return positions[start:end]
    return positions[:, start:end]


def _mark_step_before_host_read(tensor: torch.Tensor | None) -> None:
    """Flush the lazy graph before a device→host read of ``tensor``.

    H23-2(a): only a *device* read needs the guard. A CPU twin (``control``)
    is read without touching the HPU queue, and calling ``mark_step`` for it
    would split the lazy graph once per QSA layer for nothing. Eager mode is
    already a no-op inside ``lazy_mark_step``. With the switch off the guard
    runs unconditionally, exactly as before H23.
    """
    if not _h23_enabled("spans"):
        _lazy_mark_step()
        return
    if tensor is None or tensor.device.type != "hpu":
        return
    _lazy_mark_step()


def _chunk_start(positions: torch.Tensor) -> int:
    # Lazy mode: flush the pending graph before this host read. A bare
    # .item() forces a *partial* execution -- ops already enqueued but not
    # ancestral to the read (the preceding HC/QKV segment) can be left
    # unexecuted or sliced into an invalid graph (synStatus 26 /
    # validateBPGraph). Marking first executes them in queue order, the
    # same guard packed_moe uses before its host cap read. No-op eager.
    _mark_step_before_host_read(positions)
    if positions.ndim == 1:
        return int(positions[0].item())
    return int(positions[0, 0].item())


def _compute_step_spans(
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Request spans (bucket padding trimmed) and their chunk starts.

    One host transfer for the whole step: ``positions`` is read once as a
    Python list, and both the per-span valid counts and the per-span chunk
    starts are derived from it. The HPU guard runs only when the tensor being
    read really lives on the device (H23-2(a)).
    """
    _mark_step_before_host_read(positions)
    spans = _request_spans(num_tokens, query_start_loc)
    if positions is None:
        return spans, [None] * len(spans)  # type: ignore[list-item]
    # The row that carries token positions: dim 0 for [N], row 0 of [3, N]
    # MRoPE (the temporal axis, which is what the cache index follows).
    rows = positions if positions.ndim == 1 else positions[0]
    values = rows.tolist()
    valid_prefix = [0]
    for value in values:
        valid_prefix.append(valid_prefix[-1] + (1 if value >= 0 else 0))
    trimmed: list[tuple[int, int]] = []
    starts: list[int] = []
    for start, end in spans:
        valid = valid_prefix[min(end, len(valid_prefix) - 1)] - valid_prefix[min(start, len(valid_prefix) - 1)]
        if valid > 0:
            trimmed.append((start, start + valid))
            starts.append(int(values[start]) if start < len(values) else 0)
    return trimmed, starts


def _valid_spans(
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> list[tuple[int, int]]:
    """Request spans with runner bucket padding trimmed.

    The HPU runner pads each request span to ``target_seq`` and whole batch
    slots to ``target_bs``, filling both ``positions`` and ``input_ids`` with
    ``-1`` after the real tokens. Real tokens are therefore a prefix of every
    span, and an all-pad span is skipped entirely: committing its rows would
    raise a cache-gap error, and a pad tail inside a real span would append
    garbage to the software cache.

    H23-2(b): the extraction is cached per step (see
    ``control.cached_span_plan``), so the twelve QSA layers of a decode step
    share one host read instead of taking one each.
    """
    return _step_spans(num_tokens, query_start_loc, positions)[0]


def _step_spans(
    num_tokens: int,
    query_start_loc: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """``(spans, chunk_starts)`` for this step, extracted at most once."""
    if not _h23_enabled("spans"):
        return _compute_step_spans(num_tokens, query_start_loc, positions)
    plan = _cached_span_plan(
        num_tokens,
        query_start_loc,
        positions,
        lambda: _compute_step_spans(num_tokens, query_start_loc, positions),
    )
    return plan.spans, plan.starts


def _tile_replay_enabled() -> bool:
    """One switch for the whole Qwen3.8-style graph-size compromise.

    H23-3: the window exists for *recipe stability* only. In eager mode
    (``PT_HPU_LAZY_MODE=0``) there is no recipe to stabilize, so padding the
    pooled cache to a 2048-block bundle and sorting all of it every decode
    token is pure waste -- eager uses the unpadded pooled cache directly.
    """
    from vllm_gaudi.extension import prefill_tile_replay

    if not prefill_tile_replay.enabled():
        return False
    return _is_hpu_lazy_mode() or not _h23_enabled("tile")


# Backing-buffer growth quanta for _QSASlot: token rows (raw K / K / V /
# positions) grow 512 at a time, pooled indexer blocks 128 at a time, so a
# decode step amortizes to a reallocation every 512 tokens / 128 blocks
# instead of a full-history ``torch.cat`` copy on every token.
# H30: capacity grows in 4096-token steps so the compiled decode attention, which reads the
# capacity buffers, is recompiled once per 4096 tokens rather than per 512.
#: ``VLLM_QWEN4EXP_QSA_TIMING=1`` (diagnostics, synchronizes): card time of the prefill QSA stages.
QSA_STAGE_TIMES: dict[str, float] = {}


class _StageTimer:
    def __init__(self, name: str, device: torch.device) -> None:
        self.name = name
        self.on = device.type == "hpu" and os.environ.get("VLLM_QWEN4EXP_QSA_TIMING", "0") == "1"

    def __enter__(self):
        if self.on:
            import time

            torch.hpu.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.on:
            import time

            torch.hpu.synchronize()
            QSA_STAGE_TIMES[self.name] = QSA_STAGE_TIMES.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


_TOKEN_CHUNK = 4096
_BLOCK_CHUNK = 1024
# H31: past 32K tokens the capacity steps get coarser, so a 262K context has 14 buffer shapes (and
# 14 decode graphs to compile at warmup) instead of 64. Padding stays under 50%.
_TOKEN_CAPACITIES = tuple(range(4096, 32768 + 1, 4096)) + (49152, 65536, 98304, 131072, 196608, 262144)


def token_capacity_for(needed: int) -> int:
    """Rows a token buffer is allocated with to hold ``needed`` tokens."""
    for capacity in _TOKEN_CAPACITIES:
        if needed <= capacity:
            return capacity
    return ((needed + _TOKEN_CHUNK - 1) // _TOKEN_CHUNK) * _TOKEN_CHUNK


def token_capacities_upto(max_tokens: int) -> list[int]:
    """Every token capacity a request of at most ``max_tokens`` tokens can reach."""
    out = [c for c in _TOKEN_CAPACITIES if c < max_tokens]
    out.append(token_capacity_for(max_tokens))
    return sorted(set(out))


class _QSASlot:
    """Per-slot QSA cache state over pre-allocated backing buffers (H12).

    Each token tensor (``raw_keys``, ``key``, ``value``, ``rope_positions``)
    tracks its own active length (they move together in normal operation,
    but a caller truncating one attribute must not shrink the others), and
    ``pooled`` (completed indexer blocks, normed + rotated ``[blocks, D]``)
    has its own. Each public attribute is a *view* of its active prefix
    ``[0:length]`` of a larger backing buffer, so in-place writes through a
    view (e.g. ``state.key[:-1].zero_()``) land in the cache.

    Appends write in place via slice assignment and only reallocate —
    copying the active prefix once — when capacity is exceeded. Direct
    assignment (``state.key = tensor``, including truncating views such as
    ``state.key = state.key[:-1]``) is supported through the property
    setters: it copies the tensor into the backing buffer and resizes the
    active prefix, so existing callers and unit tests that read, mutate, or
    replace these attributes keep working unchanged.
    """

    def __init__(
        self,
        raw_keys: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_positions: torch.Tensor,
        pooled: torch.Tensor,
    ) -> None:
        self._buf_raw_keys: torch.Tensor | None = None
        self._buf_key: torch.Tensor | None = None
        self._buf_value: torch.Tensor | None = None
        self._buf_rope_positions: torch.Tensor | None = None
        self._buf_pooled: torch.Tensor | None = None
        self._len_raw_keys = 0
        self._len_key = 0
        self._len_value = 0
        self._len_rope_positions = 0
        self._pooled_len = 0
        self.raw_keys = raw_keys
        self.key = key
        self.value = value
        self.rope_positions = rope_positions
        self.pooled = pooled

    # -- buffer plumbing -------------------------------------------------
    @staticmethod
    def _seq_axis(tensor: torch.Tensor) -> int:
        # Token axis: dim 0 for [N, ...] and [N]; dim 1 for [3, N] MRoPE rows.
        return 0 if tensor.ndim <= 1 else 1

    @staticmethod
    def _capacity(needed: int, chunk: int) -> int:
        if chunk == _TOKEN_CHUNK:
            return token_capacity_for(needed)
        if chunk == _BLOCK_CHUNK:
            # Pooled blocks follow the token capacities (one block per _TOKEN_CHUNK // _BLOCK_CHUNK tokens).
            per = _TOKEN_CHUNK // _BLOCK_CHUNK
            return token_capacity_for(needed * per) // per
        return max(chunk, ((needed + chunk - 1) // chunk) * chunk)

    @staticmethod
    def _prefix(tensor: torch.Tensor, length: int, axis: int) -> torch.Tensor:
        return tensor[:length] if axis == 0 else tensor[:, :length]

    @staticmethod
    def _window(tensor: torch.Tensor, start: int, end: int, axis: int) -> torch.Tensor:
        return tensor[start:end] if axis == 0 else tensor[:, start:end]

    def _grow_buffer(
        self,
        attr: str,
        template: torch.Tensor,
        needed: int,
        active: int,
        chunk: int,
        axis: int,
    ) -> torch.Tensor:
        """Return a buffer holding the ``active`` prefix, capacity >= needed."""
        buf = getattr(self, attr)
        shape = list(template.shape)
        compatible = (
            buf is not None
            and buf.dtype == template.dtype
            and buf.device == template.device
            and buf.ndim == template.ndim
            and all(buf.shape[i] == shape[i] for i in range(template.ndim) if i != axis)
        )
        if compatible and buf.shape[axis] >= needed:
            return buf
        shape[axis] = self._capacity(needed, chunk)
        new_buf = template.new_empty(tuple(shape))
        with torch.no_grad():
            if compatible and active > 0:
                self._window(new_buf, 0, active, axis).copy_(
                    self._prefix(buf, active, axis))
            # Zero the unused capacity. Static-bucket lazy attention
            # (dense_prefix_attention(valid_num_keys=...)) runs GEMMs over
            # the whole backing buffer and masks these rows only *after*
            # the score GEMM, so uninitialized rows could otherwise leak
            # Inf/NaN bit patterns into a 0 * NaN product. This runs only
            # on (re)allocation -- every _TOKEN_CHUNK/_BLOCK_CHUNK rows.
            if shape[axis] > active:
                self._window(new_buf, active, shape[axis], axis).zero_()
        setattr(self, attr, new_buf)
        return new_buf

    _LEN_ATTR = {
        "_buf_raw_keys": "_len_raw_keys",
        "_buf_key": "_len_key",
        "_buf_value": "_len_value",
        "_buf_rope_positions": "_len_rope_positions",
        "_buf_pooled": "_pooled_len",
    }

    def _set_length(self, attr: str, length: int) -> None:
        setattr(self, self._LEN_ATTR[attr], length)

    def _length(self, attr: str) -> int:
        return getattr(self, self._LEN_ATTR[attr])

    def _assign(self, attr: str, tensor: torch.Tensor, axis: int, chunk: int) -> None:
        """Replace an attribute's active content with ``tensor`` (a copy)."""
        buf = getattr(self, attr)
        length = tensor.shape[axis] if tensor.ndim else 0
        if buf is not None and tensor.ndim == buf.ndim and tensor.storage_offset() == 0:
            try:
                same_storage = (tensor.untyped_storage().data_ptr()
                                == buf.untyped_storage().data_ptr())
            except Exception:  # noqa: BLE001 - empty tensors may lack storage
                same_storage = False
            # Assigning back a prefix view of our own buffer (the rollback
            # idiom ``state.key = state.key[:-1]``): just resize that one
            # attribute's prefix; the other attributes keep their lengths.
            if same_storage and all(
                    tensor.shape[i] == buf.shape[i] for i in range(tensor.ndim) if i != axis):
                self._set_length(attr, length)
                return
        buf = self._grow_buffer(attr, tensor, length, active=0, chunk=chunk, axis=axis)
        if length:
            with torch.no_grad():
                self._prefix(buf, length, axis).copy_(tensor.detach())
        self._set_length(attr, length)

    def _append(self, attr: str, tensor: torch.Tensor, axis: int, chunk: int) -> None:
        n = tensor.shape[axis]
        if n == 0:
            return
        old = self._length(attr)
        buf = self._grow_buffer(attr, tensor, old + n, active=old, chunk=chunk, axis=axis)
        with torch.no_grad():
            self._window(buf, old, old + n, axis).copy_(tensor.detach())
        self._set_length(attr, old + n)

    # -- public token views ----------------------------------------------
    @property
    def seq_len(self) -> int:
        """Active token count, canonically the indexer raw-key length."""
        return self._len_raw_keys

    @property
    def raw_keys(self) -> torch.Tensor:
        return self._prefix(self._buf_raw_keys, self._len_raw_keys, 0)

    @raw_keys.setter
    def raw_keys(self, tensor: torch.Tensor) -> None:
        self._assign("_buf_raw_keys", tensor, 0, _TOKEN_CHUNK)

    @property
    def key(self) -> torch.Tensor:
        return self._prefix(self._buf_key, self._len_key, 0)

    @key.setter
    def key(self, tensor: torch.Tensor) -> None:
        self._assign("_buf_key", tensor, 0, _TOKEN_CHUNK)

    # Task-facing aliases: the main KV cache is ``key``/``value`` here.
    @property
    def k(self) -> torch.Tensor:
        return self.key

    @k.setter
    def k(self, tensor: torch.Tensor) -> None:
        self.key = tensor

    @property
    def value(self) -> torch.Tensor:
        return self._prefix(self._buf_value, self._len_value, 0)

    @value.setter
    def value(self, tensor: torch.Tensor) -> None:
        self._assign("_buf_value", tensor, 0, _TOKEN_CHUNK)

    @property
    def v(self) -> torch.Tensor:
        return self.value

    @v.setter
    def v(self, tensor: torch.Tensor) -> None:
        self.value = tensor

    @property
    def rope_positions(self) -> torch.Tensor:
        buf = self._buf_rope_positions
        return self._prefix(buf, self._len_rope_positions, self._seq_axis(buf))

    @rope_positions.setter
    def rope_positions(self, tensor: torch.Tensor) -> None:
        self._assign("_buf_rope_positions", tensor, self._seq_axis(tensor), _TOKEN_CHUNK)

    @property
    def pooled(self) -> torch.Tensor:
        return self._prefix(self._buf_pooled, self._pooled_len, 0)

    @pooled.setter
    def pooled(self, tensor: torch.Tensor) -> None:
        self._assign("_buf_pooled", tensor, 0, _BLOCK_CHUNK)

    # -- static-capacity views (lazy-mode decode) --------------------------
    @property
    def token_capacity(self) -> int:
        """Backing capacity (rows) of the main KV buffers, >= seq_len."""
        return 0 if self._buf_key is None else int(self._buf_key.shape[0])

    @property
    def raw_keys_backing(self) -> torch.Tensor:
        """Full-capacity indexer raw-key buffer (active prefix is ``raw_keys``)."""
        return self._buf_raw_keys

    @property
    def key_backing(self) -> torch.Tensor:
        """Full-capacity key buffer (active prefix is ``key``)."""
        return self._buf_key

    @property
    def value_backing(self) -> torch.Tensor:
        """Full-capacity value buffer (active prefix is ``value``)."""
        return self._buf_value

    def truncate_tokens(self, length: int) -> None:
        """Roll every token attribute back to ``length`` rows in place.

        Length-only (same-storage prefix) assignment, so no device copy:
        the next append simply overwrites from ``length``. Used by
        steady-state decode benchmarking to hold S_kv at the primed
        length instead of growing it on every benchmarked token. Pooled
        blocks are *not* touched here (their count depends on the
        compress ratio); callers roll them back via :meth:`truncate_pooled`.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        for attr, current in (
                ("raw_keys", self._len_raw_keys), ("key", self._len_key),
                ("value", self._len_value)):
            if current > length:
                setattr(self, attr, getattr(self, attr)[:length])
        if self._len_rope_positions > length:
            rope = self.rope_positions
            axis = self._seq_axis(rope)
            setattr(self, "rope_positions",
                    rope[:length] if axis == 0 else rope[:, :length])

    def truncate_pooled(self, blocks: int) -> None:
        """Roll the pooled indexer cache back to ``blocks`` rows."""
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        if self._pooled_len > blocks:
            self.pooled = self.pooled[:blocks]

    # -- in-place appends (the decode hot path: no torch.cat) -------------
    def append_tokens(
        self,
        raw_keys: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> None:
        """Append one chunk of token rows in place, growing rarely."""
        if raw_keys.shape[0] == 0:
            return
        self._append("_buf_raw_keys", raw_keys, 0, _TOKEN_CHUNK)
        self._append("_buf_key", key, 0, _TOKEN_CHUNK)
        self._append("_buf_value", value, 0, _TOKEN_CHUNK)
        self._append(
            "_buf_rope_positions", rope_positions,
            self._seq_axis(rope_positions), _TOKEN_CHUNK)

    def append_pooled(self, pooled: torch.Tensor) -> None:
        """Append newly completed indexer blocks in place."""
        self._append("_buf_pooled", pooled, 0, _BLOCK_CHUNK)

    def reserve_pooled(self, blocks: int) -> None:
        """Grow the pooled buffer to ``blocks`` rows now, together with the token buffers.

        H31: otherwise it grows a few decode tokens after them, and the decode graph sees one extra
        (token capacity, pooled capacity) pair per growth.
        """
        if self._buf_pooled is None or self._buf_pooled.shape[0] >= blocks:
            return
        self._grow_buffer("_buf_pooled", self._buf_pooled, blocks, self._pooled_len, _BLOCK_CHUNK, 0)


# The task-facing name for the per-slot cache state above.
QSACacheState = _QSASlot


class QSAIndexer(nn.Module):
    """Replicated indexer projection. Names match the checkpoint prefix."""

    def __init__(self, config, prefix: str = "") -> None:
        super().__init__()
        if int(config.indexer_kv_heads) != 1:
            raise ValueError("Qwen4Exp QSA requires indexer_kv_heads=1")
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.prefix = prefix
        proj = f"{prefix}.index_qk_proj" if prefix else "index_qk_proj"
        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            prefix=proj,
        )
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=eps)

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Norm+rotate Q; return raw (unrotated) K of shape ``[T, D]``."""
        if hidden_states.device.type == "hpu" and _is_hpu_lazy_mode():
            # Lazy HPU only: project in fp32 (BF16 inputs, fp32 accumulate,
            # one BF16 rounding -- the accumulation the CPU/eager reference
            # GEMM uses). The lazy-compiled BF16 GEMM rounds differently
            # enough that the pooled indexer rows flip by 1 ULP, and the
            # near-tied block scores then pick different blocks than the
            # reference (one swapped block carried every over-tolerance
            # output element in the layer test). Eager keeps the stock
            # linear, bit-identical to before.
            qk = torch.nn.functional.linear(
                hidden_states.float(), self.index_qk_proj.weight.float()).to(hidden_states.dtype)
        else:
            qk = _linear_out(self.index_qk_proj, hidden_states)
        return self.project_qk_from_proj(qk, positions, rotary_emb)

    def project_qk_from_proj(
        self,
        qk: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Norm+rotate an already projected ``[T, (n_heads + kv) * D]`` QK block.

        Split out for H23-4: when the caller merged ``qkv_proj`` and
        ``index_qk_proj`` into one GEMM, this block arrives as a column slice
        of that GEMM's output. Strided row views are handled in place (the HPU
        fused RMSNorm and the elementwise RoPE both accept them), so no
        ``contiguous()`` copy is added.
        """
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.view(-1, self.index_n_heads, self.index_head_dim)
        q = gemma_rmsnorm(q, self.q_layernorm.weight, self.q_layernorm.variance_epsilon)
        cos_sin = _cos_sin_rows(rotary_emb, positions, q)
        q = apply_partial_rope(q, cos_sin)
        # Materialize both outputs (lazy only; plain contiguous in eager):
        # they are consumed by the cache append and the selection scores in
        # the same forward, and segment-internal the compiler would fuse
        # the indexer GEMM/norm/rope chain into those consumers without
        # the BF16 boundaries (see qsa_math.apply_partial_rope).
        return _materialized(q), _materialized(token_k.reshape(-1, self.index_head_dim))


def _cos_sin_rows(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    like: torch.Tensor,
) -> torch.Tensor:
    """``[N, rotary_dim]`` cos||sin rows for ``positions``.

    H28-2: the composition is a property of the *step* (same positions, same
    cache, same sections for all twelve QSA layers), so it runs once per step
    and the main q/k RoPE shares the result with the indexer (see
    :func:`_apply_qk_rope`). ``VLLM_QWEN4EXP_QSA_H28_ROPE=0`` restores the
    per-call composition.
    """
    cache = rotary_emb.cos_sin_cache
    cast = cache.device != like.device or cache.dtype != like.dtype
    if cast:
        cache = cache.to(device=like.device, dtype=like.dtype)
    section = getattr(rotary_emb, "mrope_section", None)
    interleaved = bool(getattr(rotary_emb, "mrope_interleaved", False))

    def compose() -> torch.Tensor:
        return mrope_cos_sin(
            cache,
            positions,
            mrope_section=section,
            mrope_interleaved=interleaved,
        )

    if cast or not _h28_enabled("rope"):
        # A cast cache is a fresh temporary per call, so it has no stable
        # identity to key a step cache on; it also only happens off HPU.
        return compose()
    # The cache identity, not its ``data_ptr()``: a lazy HPU tensor's data
    # pointer is not a free read (it can force a graph flush), and the buffer
    # object itself is stable for the module's lifetime -- ``nn.Module``
    # buffers live in ``_buffers`` and are only replaced by a dtype/device cast.
    key = (
        id(cache),
        cache.dtype,
        cache.device,
        tuple(cache.shape),
        tuple(section or ()),
        interleaved,
        tuple(positions.shape),
    )
    return _cached_mrope_cos_sin(positions, key, compose)


def _apply_qk_rope(
    rotary_emb: nn.Module,
    cos_sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``MRotaryEmbedding.forward_native``'s apply step on shared cos/sin rows.

    ``self.rotary_emb(positions, q, k)`` composes the M-RoPE cos/sin and then
    rotates; only the composition repeats across layers, the rotation is the
    work itself. Feeding the step's rows through the same
    ``apply_rotary_emb`` the vendor path uses keeps the result bit-identical
    (``mrope_cos_sin`` reproduces ``cos_sin_cache[positions]`` + ``chunk`` +
    interleave), and the view/slice/cat tail is the vendor tail spelled out.
    ``cos_sin`` is ``[N, rotary_dim]`` with cos then sin concatenated.
    """
    cos, sin = cos_sin.chunk(2, dim=-1)
    head_size = rotary_emb.head_size
    rotary_dim = rotary_emb.rotary_dim
    rotated = []
    for x in (q, k):
        shape = x.shape
        x = x.view(shape[0], -1, head_size)
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = rotary_emb.apply_rotary_emb.forward_native(x_rot, cos, sin)
        rotated.append(torch.cat((x_rot, x_pass), dim=-1).reshape(shape))
    return rotated[0], rotated[1]


class Qwen4ExpQSAAttention(Qwen3NextAttention):
    """Full attention with a QSA index side branch (torch bring-up)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.layer_id = int(layer_id)
        self.indexer = QSAIndexer(config, prefix=f"{prefix}.indexer" if prefix else "indexer")
        # Not a buffer: must not appear in state_dict or the weight loader.
        self._bringup: list[_QSASlot | None] = []
        # H23-4 merge cache (see _merged_projection_weight). Registered here so
        # a plain load_state_dict (tests, offline layer builds) also drops it.
        self._merged_projection_cache = None
        self._qkv_rows = 0
        self.register_load_state_dict_post_hook(
            lambda module, incompatible_keys: module.invalidate_h23_merge())

    def reset_bringup_cache(self) -> None:
        self._bringup.clear()

    # -- H23-4: fused QK RMSNorm + RoPE -----------------------------------
    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``Qwen3NextAttention._project_qkv_gate`` with the fused HPU RMSNorm.

        Identical arithmetic to the vendor eager branch (split q/gate/k/v,
        Gemma RMSNorm over ``head_dim``, then the model's MRoPE), except that
        the two head-dim norms run as one ``torch.ops.hpu.rms_norm`` call each
        (``addOneToWeight=True`` is the Gemma ``(1 + w)`` affine) instead of
        the decomposed ``ir.ops.rms_norm`` chain: 4.5 ms vs 1.1 ms per
        12-layer decode token in ``bench_h23_qsa``'s micro attribution. The
        vendor path is taken verbatim whenever the fused op declines (switch
        off, non-HPU, non-BF16, CPU tests) -- so the CPU reference numerics are
        unchanged.
        """
        if not _h23_enabled("fuse"):
            return super()._project_qkv_gate(qkv, positions)
        q_offset, q_width = 0, self.q_size * (2 if self.attn_output_gate else 1)
        k_offset = q_width
        fused_q = hpu_rms_norm(
            self._qk_norm_view(qkv, q_offset, q_width, self.num_heads,
                               gated=self.attn_output_gate),
            self.q_norm.weight, self.q_norm.variance_epsilon)
        fused_k = hpu_rms_norm(
            self._qk_norm_view(qkv, k_offset, self.kv_size, self.num_kv_heads,
                               gated=False),
            self.k_norm.weight, self.k_norm.variance_epsilon)
        if fused_q is None or fused_k is None:
            return super()._project_qkv_gate(qkv, positions)
        return self._rope_qkv_gate(qkv, positions, fused_q, fused_k)

    def _qk_norm_view(
        self,
        qkv: torch.Tensor,
        offset: int,
        width: int,
        heads: int,
        *,
        gated: bool,
    ) -> torch.Tensor:
        """The ``[T, heads, head_dim]`` slice of ``qkv`` the norm applies to.

        With output gating the q block carries ``q || gate`` interleaved per
        head, so the q half is ``block.view(T, heads, 2 * head_dim)[..., :D]``
        -- exactly the vendor ``chunk``-then-reshape layout. k has no gate.
        """
        num_tokens = qkv.shape[0]
        block = qkv[:, offset:offset + width]
        if gated:
            return block.view(num_tokens, heads, 2 * self.head_dim)[..., : self.head_dim]
        return block.view(num_tokens, heads, self.head_dim)

    def _rope_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_normed: torch.Tensor,
        k_normed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Vendor tail of ``_project_qkv_gate``: rope + gate layout."""
        num_tokens = qkv.shape[0]
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            gate = q_gate.view(num_tokens, self.num_heads, 2 * self.head_dim)
            gate = gate[..., self.head_dim:].reshape(num_tokens, -1)
        else:
            _q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
        q = q_normed.reshape(num_tokens, self.num_heads * self.head_dim)
        k = k_normed.reshape(num_tokens, self.num_kv_heads * self.head_dim)
        if _h28_enabled("rope"):
            # H28-2: compose the M-RoPE rows once per step (the indexer below
            # asks for the same rows) and rotate with them.
            cos_sin = _cos_sin_rows(self.rotary_emb, positions, q)
            q, k = _apply_qk_rope(self.rotary_emb, cos_sin, q, k)
        else:
            q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    # -- H23-4: merged qkv + index_qk projection --------------------------
    def invalidate_h23_merge(self) -> None:
        """Drop the merged projection weight (after a weight (re)load)."""
        self._merged_projection_cache = None
        self._qkv_rows = 0

    def _merged_projection_weight(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """``[qkv_rows + index_rows, hidden]`` concatenation, or None to decline.

        ``qkv_proj`` and ``index_qk_proj`` take the *same* ``hidden_states`` and
        are both plain BF16 linears after the ``qwen4exp_plain`` load, so one
        GEMM over the concatenated weight replaces two launches and one of the
        two full weight reads (measured 44.3 -> 34.2 us at T=1, 47.8 -> 34.5 us
        at T=4: the merged read is the same 66 MiB either way, so the decode
        GEMM becomes bandwidth-bound rather than launch-bound).

        Declines -- falling back to the two stock linears -- when the switch is
        off, in lazy mode (the indexer projection has a deliberate fp32 lazy
        path, see ``QSAIndexer.project_qk``), during compile-only warmup (a
        tensor produced there has no data and must not be cached), off HPU, for
        non-BF16 or non-plain (quantized) weights, or when the two projections
        disagree about the input width.

        The merged tensor is cached and rebuilt only when either weight's
        storage or version changes (the same guard as
        ``hyperconnection_amd._gamma2d``), so the steady-state cost is zero.
        """
        if not (_h23_enabled("fuse") and _h23_enabled("merge")):
            return None
        if _is_hpu_lazy_mode() or _in_compile_only_mode():
            return None
        if hidden_states.device.type != "hpu" or hidden_states.dtype != torch.bfloat16:
            return None
        qkv_w = getattr(self.qkv_proj, "weight", None)
        idx_w = getattr(self.indexer.index_qk_proj, "weight", None)
        if qkv_w is None or idx_w is None:
            return None
        if qkv_w.dtype != torch.bfloat16 or idx_w.dtype != torch.bfloat16:
            return None
        if qkv_w.device.type != "hpu" or idx_w.device.type != "hpu":
            return None
        if not (_is_plain_linear(self.qkv_proj) and _is_plain_linear(self.indexer.index_qk_proj)):
            return None
        if qkv_w.ndim != 2 or idx_w.ndim != 2 or qkv_w.shape[1] != idx_w.shape[1]:
            return None
        if hidden_states.shape[-1] != qkv_w.shape[1]:
            return None
        key = (
            qkv_w.data_ptr(), qkv_w._version, idx_w.data_ptr(), idx_w._version,
            tuple(qkv_w.shape), tuple(idx_w.shape), qkv_w.device, qkv_w.dtype,
        )
        cached = getattr(self, "_merged_projection_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        from .merged_weights import views_of

        if cached is not None and views_of(cached[1], qkv_w, idx_w):
            # The parameters are views of the cached concatenation; only its address moved.
            self._merged_projection_cache = (key, cached[1])
            return cached[1]
        qkv_rows = int(qkv_w.shape[0])
        from .merged_weights import merge_rows

        merged = merge_rows(self.qkv_proj.weight, self.indexer.index_qk_proj.weight)
        if self.qkv_proj.weight.data_ptr() == merged.data_ptr():
            # H31: both parameters are now views of ``merged``; key the cache on them.
            qkv_w = self.qkv_proj.weight
            idx_w = self.indexer.index_qk_proj.weight
            key = (
                qkv_w.data_ptr(), qkv_w._version, idx_w.data_ptr(), idx_w._version,
                tuple(qkv_w.shape), tuple(idx_w.shape), qkv_w.device, qkv_w.dtype,
            )
        self._qkv_rows = qkv_rows
        self._merged_projection_cache = (key, merged)
        return merged

    def _commit(
        self,
        slot: int,
        rope_positions: torch.Tensor,
        raw_keys: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start: int | None = None,
    ) -> torch.Tensor:
        """Append one contiguous chunk. Returns logical positions ``[n]``."""
        while len(self._bringup) <= slot:
            self._bringup.append(None)
        # No detach().contiguous() copies here: the slot copies each chunk
        # once, in place, into its pre-allocated backing buffers (which also
        # detaches it from the autograd graph, as the old copies did).
        n = raw_keys.shape[0]
        start = _chunk_start(rope_positions) if start is None else int(start)
        state = self._bringup[slot] if slot < len(self._bringup) else None
        have = 0 if state is None else state.raw_keys.shape[0]
        if os.environ.get("VLLM_QWEN4EXP_QSA_DEBUG", "") == "1":
            print(f"[qsa-commit] slot={slot} have={have} start={start} n={n}",
                  flush=True)
        # A chunk that does not continue the cached history is a cold start:
        # the runner's warmup dummies carry a synthetic context for which no
        # tokens ever passed through this software cache, and a real request
        # always begins at position 0. Rebuild the slot instead of failing;
        # a silent gap would otherwise be indistinguishable from a bug.
        if start == 0 or start != have:
            if start != 0 and not getattr(self, "_gap_warned", False):
                logger.warning(
                    "QSA cache cold start on slot %d (cached %d, chunk at %d); "
                    "history is discarded", slot, have, start)
                self._gap_warned = True
            state = _QSASlot(
                raw_keys,
                key,
                value,
                rope_positions,
                raw_keys.new_empty(0, self.indexer.index_head_dim),
            )
            while len(self._bringup) <= slot:
                self._bringup.append(None)
            self._bringup[slot] = state
        else:
            state.append_tokens(raw_keys, key, value, rope_positions)
        if os.environ.get("VLLM_QWEN4EXP_QSA_POOL_AT", "1") != "0":
            state.reserve_pooled(state.token_capacity // self.indexer.compress_ratio)
        # Selection reads the pooled rows only above the token budget. Pooling below it costs about
        # 50 launches per layer on every fourth decode token and its result would go unused.
        if os.environ.get("VLLM_QWEN4EXP_QSA_LAZY_POOL", "1") == "0" or state.raw_keys.shape[0] > self.indexer.token_topk:
            self._pool_new_blocks(state)
        # Keys are stored at cache indices 0..S-1, and the indexer treats the
        # returned positions as those indices. A cold start discarded any
        # history, so this chunk occupies 0..n-1 even when the runner's raw
        # position is a synthetic context (warmup dummies start at 6144).
        # Continuation already has start == have, which is the cache index.
        logical_start = 0 if start == 0 or start != have else have
        return torch.arange(logical_start, logical_start + n, device=raw_keys.device)

    def _pool_new_blocks(self, state: _QSASlot) -> None:
        """Incrementally pool newly completed indexer blocks.

        The pooled cache is what the shape-stable selection reads: its width
        only grows, old rows never change, and blocks completing in this chunk
        are pooled exactly once. This is the QSA counterpart of the fixed
        attention tile in the Qwen3.8 graph-size compromise.
        """
        ratio = self.indexer.compress_ratio
        total_blocks = state.raw_keys.shape[0] // ratio
        old_blocks = state.pooled.shape[0]
        if total_blocks <= old_blocks:
            return
        groups = state.raw_keys[old_blocks * ratio : total_blocks * ratio]
        starts = torch.arange(old_blocks, total_blocks, device=groups.device) * ratio
        rope = state.rope_positions
        block_pos = rope.index_select(0, starts) if rope.ndim == 1 else rope[:, starts]
        cos_sin = _cos_sin_rows(self.rotary_emb, block_pos, state.raw_keys)
        block_keys = groups.view(total_blocks - old_blocks, ratio, self.indexer.index_head_dim)
        from . import decode_graph

        if total_blocks - old_blocks == 1 and decode_graph.enabled() and not decode_graph.verifying():
            # H30: the one block a decode step completes, as one graph instead of ~45 launches.
            if os.environ.get("VLLM_QWEN4EXP_QSA_POOL_AT", "1") != "0":
                # H31: index the capacity buffer instead of passing an offset view (see pool_block_at).
                token_index = torch.arange(old_blocks * ratio, total_blocks * ratio, device=groups.device)
                decode_graph.warm_pool_block_at(state.raw_keys_backing, token_index, self.indexer.k_layernorm.weight,
                                                float(self.indexer.k_layernorm.variance_epsilon), cos_sin,
                                                token_capacities_upto(decode_graph.warm_max_tokens()))
                pooled = decode_graph.pool_block_at(state.raw_keys_backing, token_index,
                                                    self.indexer.k_layernorm.weight,
                                                    float(self.indexer.k_layernorm.variance_epsilon), cos_sin)
            else:
                pooled = decode_graph.pool_block(block_keys, self.indexer.k_layernorm.weight,
                                                 float(self.indexer.k_layernorm.variance_epsilon), cos_sin)
        else:
            pooled = pool_indexer_blocks(
                block_keys,
                k_norm_weight=self.indexer.k_layernorm.weight,
                eps=self.indexer.k_layernorm.variance_epsilon,
                block_cos_sin=cos_sin,
            )
        state.append_pooled(pooled)

    def _windowed_indices(
        self,
        q: torch.Tensor,
        state: _QSASlot,
        logical: torch.Tensor,
        host_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Selection over a compile-stable bundled window of pooled blocks.

        The window width is ``bundle_count * context_tile`` blocks — the
        eleven prefill shapes from the tile-replay plan — and never the raw
        context length, so the indexer scoring keeps a fixed compiled program
        while the host picks the bundle from the real block count.
        """
        from vllm_gaudi.extension import prefill_tile_replay

        ratio = self.indexer.compress_ratio
        blocks_per_tile = prefill_tile_replay.context_tile() // ratio
        total = state.pooled.shape[0]
        n_tiles = (total + blocks_per_tile - 1) // blocks_per_tile
        width = prefill_tile_replay.bundle_count(max(n_tiles, 1)) * blocks_per_tile
        window = state.pooled
        if window.shape[0] < width:
            window = torch.cat((window, window.new_zeros(width - window.shape[0], window.shape[1])))
        return select_qsa_indices_from_pooled(
            q,
            window,
            logical,
            total_blocks=total,
            token_budget=self.indexer.token_topk,
            compress_ratio=ratio,
            host_positions=host_positions,
        )

    def _block_cos_sin(self, state: _QSASlot, like: torch.Tensor) -> torch.Tensor | None:
        ratio = self.indexer.compress_ratio
        num_blocks = state.raw_keys.shape[0] // ratio
        if num_blocks == 0:
            return None
        starts = torch.arange(num_blocks, device=state.raw_keys.device) * ratio
        rope = state.rope_positions
        block_pos = rope.index_select(0, starts) if rope.ndim == 1 else rope[:, starts]
        return _cos_sin_rows(self.rotary_emb, block_pos, like)

    def _attend_cached(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_idx: torch.Tensor,
        raw_k: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor | None,
    ) -> torch.Tensor:
        """Commit this step's keys to the software cache and attend over it. ``[T, Hq, D]``."""
        num_tokens = q.shape[0]
        # Zeros, not empty: pad rows flow onward and must stay finite.
        attn = q.new_zeros(num_tokens, self.num_heads, self.head_dim)
        from .control import cpu_twin

        cpu_pos = cpu_twin(positions, "positions")
        cpu_qsl = cpu_twin(query_start_loc, "query_start_loc")
        span_pos = cpu_pos if cpu_pos is not None else positions
        span_qsl = cpu_qsl if cpu_qsl is not None else query_start_loc
        # H23-2: one span/chunk-start extraction for the whole step (this is
        # the 12-layer-per-step host sync), including the chunk starts that
        # used to cost a device `.item()` per layer.
        spans, chunk_starts = _step_spans(num_tokens, span_qsl, span_pos)
        for slot, (start, end) in enumerate(spans):
            chunk_start = chunk_starts[slot] if len(chunk_starts) > slot else None
            logical = self._commit(
                slot,
                _slice_positions(positions, start, end),
                raw_k[start:end],
                k[start:end],
                v[start:end],
                start=chunk_start,
            )
            state = self._bringup[slot]
            assert state is not None
            q_chunk = q[start:end]
            k_cache = state.key
            v_cache = state.value
            s_kv = k_cache.shape[0]
            if s_kv <= self.indexer.token_topk:
                # Short-context fast path (H12): the whole active KV fits
                # the indexer budget (S_kv <= token_topk, hence also
                # total_blocks <= block_topk), so selection would keep
                # every causal token anyway. Skip select_qsa_indices and
                # the index_select/gather entirely and attend to the active
                # prefix views directly. Decode (q_len == 1) passes no
                # query_positions: its single query is at S_kv - 1, after
                # every cached key, so no causal mask is built either.
                #
                # Lazy mode (PT_HPU_LAZY_MODE=1): the active prefix length
                # grows every decode token (257, 258, ...), so slicing
                # [0:S_kv] would give Synapse a new attention shape -- and
                # a new recipe -- per token. Attend over the static
                # backing capacity instead (512-row quanta, zero-filled
                # past the active prefix) and mask positions >= S_kv in
                # dense_prefix_attention; shapes then change only when
                # the backing buffer grows. Eager keeps the exact slice
                # (zero recompile cost there). Opt out / force with
                # VLLM_QWEN4EXP_QSA_STATIC_BUCKET=0/1.
                if (qsa_static_bucket_enabled(k_cache.device.type)
                        and state.token_capacity > s_kv):
                    k_cache = state.key_backing
                    v_cache = state.value_backing
                    attn[start:end] = dense_prefix_attention(
                        q_chunk,
                        k_cache,
                        v_cache,
                        self.scaling,
                        query_positions=None if end - start == 1 else logical,
                        valid_num_keys=s_kv,
                    )
                else:
                    attn[start:end] = dense_prefix_attention(
                        q_chunk,
                        k_cache,
                        v_cache,
                        self.scaling,
                        query_positions=None if end - start == 1 else logical,
                    )
                continue
            from . import decode_graph

            if end - start == 1 and decode_graph.enabled() and not _tile_replay_enabled():
                # H30: selection + gathered attention as one graph over the capacity buffers.
                decode_graph.warm_select_attend(
                    q_chunk, q_idx[start:end], state._buf_pooled, logical, state.key_backing,
                    state.value_backing, float(self.scaling), self.indexer.compress_ratio,
                    min(self.indexer.token_topk // self.indexer.compress_ratio, state.pooled.shape[0]),
                    token_capacities_upto(decode_graph.warm_max_tokens()))
                graph_out = decode_graph.select_attend(
                    q_chunk, q_idx[start:end], state._buf_pooled, logical, state.key_backing,
                    state.value_backing, float(self.scaling), self.indexer.compress_ratio,
                    min(self.indexer.token_topk // self.indexer.compress_ratio, state.pooled.shape[0]))
                if not decode_graph.verifying():
                    attn[start:end] = graph_out
                    continue
                self._graph_select_check = graph_out
            if _tile_replay_enabled():
                indices = self._windowed_indices(q_idx[start:end], state, logical)
            else:
                # ``_commit`` already pooled every newly completed block
                # into ``state.pooled`` (incrementally, bit-identical to a
                # full re-pool; see test_qsa_layer_hpu). Select over that
                # cache directly instead of re-pooling the whole history
                # and re-deriving every block's RoPE on every step.
                #
                # H23-3: the committed chunk always occupies cache rows
                # ``S_kv - n .. S_kv - 1`` (``_commit`` rebases a cold start
                # to 0, and a continuation appends at ``have == S_kv - n``),
                # so the host knows its smallest logical position without a
                # device read and the selection can skip its -inf mask when
                # every pooled block is already visible to every query --
                # the decode case.
                n_rows = end - start
                stage = _StageTimer("select", q_chunk.device) if n_rows > 1 else None
                if stage is not None:
                    stage.__enter__()
                sel_rows = int(os.environ.get("VLLM_QWEN4EXP_QSA_SELECT_ROWS", "512") or 0)
                if (n_rows > 1 and q_chunk.device.type == "hpu" and 0 < sel_rows < n_rows
                        and state.pooled.shape[0] > int(os.environ.get("VLLM_QWEN4EXP_QSA_SELECT_TILE_MIN_BLOCKS",
                                                                       "4096"))):
                    # H31: the selection scores every query against every pooled block in fp32
                    # ([T, heads, blocks] and its relu/sum/mask copies): 3-4 GB per layer at 200K.
                    # Query-row tiles under an in-flight gate bound it; each row's selection is its own.
                    parts = []
                    gate = InflightGate(q_chunk.device)
                    for begin in range(0, n_rows, sel_rows):
                        stop = min(begin + sel_rows, n_rows)
                        gate.wait()
                        parts.append(select_qsa_indices_from_pooled(
                            q_idx[start + begin:start + stop],
                            state.pooled,
                            logical[begin:stop],
                            total_blocks=state.pooled.shape[0],
                            token_budget=self.indexer.token_topk,
                            compress_ratio=self.indexer.compress_ratio,
                            total_tokens=s_kv,
                            host_min_position=s_kv - n_rows + begin,
                        ))
                        gate.mark()
                    indices = torch.cat(parts, dim=0)
                else:
                    indices = select_qsa_indices_from_pooled(
                        q_idx[start:end],
                        state.pooled,
                        logical,
                        total_blocks=state.pooled.shape[0],
                        token_budget=self.indexer.token_topk,
                        compress_ratio=self.indexer.compress_ratio,
                        total_tokens=s_kv,
                        host_min_position=s_kv - (end - start),
                    )
                if stage is not None:
                    stage.__exit__(None, None, None)
            # Effective-budget clamping: the packed selection holds at
            # most S_kv valid entries per row, so columns past S_kv are
            # only -1 pads. Gathering the full fixed budget+ratio-1 width
            # would copy dummy row-0 keys for those pads and run attention
            # over them; slice the pads off. The select_* functions keep
            # their fixed-width return shape for direct callers.
            indices = clamp_qsa_indices(indices, s_kv)
            fsdpa_limit = int(os.environ.get("VLLM_QWEN4EXP_QSA_FSDPA_MAX_KEYS", "81920"))
            if (end - start > 1 and q_chunk.device.type == "hpu" and s_kv <= fsdpa_limit
                    and os.environ.get("VLLM_QWEN4EXP_QSA_EXACT", "").strip().lower()
                    not in ("1", "true", "yes", "on")):
                # H31: the masked-dense selection in one fused attention kernel (see fsdpa_masked_attention).
                with _StageTimer("fsdpa", q_chunk.device):
                    attn[start:end] = fsdpa_masked_attention(q_chunk, k_cache, v_cache, indices, self.scaling)
                continue
            dense_limit = int(os.environ.get("VLLM_QWEN4EXP_QSA_MASKED_DENSE_MAX_KEYS", "24576"))
            if (end - start > 1 and q_chunk.device.type == "hpu" and s_kv <= dense_limit
                    and os.environ.get("VLLM_QWEN4EXP_QSA_EXACT", "").strip().lower()
                    not in ("1", "true", "yes", "on")):
                # H30: a prefill chunk over a short cache scores every key once instead of
                # gathering its selection per query (see masked_dense_attention).
                with _StageTimer("masked_dense", q_chunk.device):
                    attn[start:end] = masked_dense_attention(q_chunk, k_cache, v_cache, indices, self.scaling)
                continue
            rows = int(os.environ.get("VLLM_QWEN4EXP_QSA_SPARSE_ROWS", "256") or 0)
            if end - start > 1 and q_chunk.device.type == "hpu" and 0 < rows < end - start:
                # H31: a 2048-row prefill chunk gathered every query's selection at once (about 9 GB of
                # K/V rows and fp32 planes). Row tiles under an in-flight gate bound that; each query row
                # is computed by itself either way.
                gate = InflightGate(q_chunk.device)
                with _StageTimer("sparse", q_chunk.device):
                    for begin in range(0, end - start, rows):
                        stop = min(begin + rows, end - start)
                        gate.wait()
                        attn[start + begin:start + stop] = sparse_attention(
                            q_chunk[begin:stop], k_cache, v_cache, indices[begin:stop], self.scaling)
                        gate.mark()
                continue
            attn[start:end] = sparse_attention(
                q_chunk,
                k_cache,
                v_cache,
                indices,
                self.scaling,
            )
            check = getattr(self, "_graph_select_check", None)
            if check is not None:
                self._graph_select_check = None
                eager = attn[start:end].float().cpu()
                delta = (check.float().cpu() - eager).abs()
                logger.warning("decode_graph verify qsa_select layer=%s s_kv=%d differ=%d/%d max=%.3e ref=%.3e",
                               getattr(self, "layer_id", "?"), s_kv, int((delta > 0).sum()), delta.numel(),
                               float(delta.max()), float(eager.abs().max()))
        return attn

    @torch._dynamo.disable  # software QSA cache keeps host-side per-slot state
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Software cache, not self.attn: the paged kernel is the HPU step.
        merged_weight = self._merged_projection_weight(hidden_states)
        index_qk = None
        if merged_weight is not None:
            # H23-4: one GEMM for qkv_proj + indexer.index_qk_proj. The two
            # column blocks are views of the merged output (no copy); the qk
            # norm / RoPE chain accepts the strided rows as they are.
            projected = nn.functional.linear(hidden_states, merged_weight)
            qkv = projected[:, : self._qkv_rows]
            index_qk = projected[:, self._qkv_rows:]
        else:
            qkv = _linear_out(self.qkv_proj, hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        if index_qk is None:
            q_idx, raw_k = self.indexer.project_qk(hidden_states, positions, self.rotary_emb)
        else:
            q_idx, raw_k = self.indexer.project_qk_from_proj(
                index_qk, positions, self.rotary_emb)

        attn = self._attend_cached(q, k, v, q_idx, raw_k, positions, query_start_loc)
        flat = attn.reshape(num_tokens, -1)
        if gate is not None:
            flat = flat * torch.sigmoid(gate)
        return _linear_out(self.o_proj, flat)
