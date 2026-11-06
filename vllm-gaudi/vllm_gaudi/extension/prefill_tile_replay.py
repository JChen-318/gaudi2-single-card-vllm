"""Host-side prefill attention tiles.

The trip count is a Python integer. Each tile is a fixed shape. This server
compiles decoder layers with ``PT_HPU_LAZY_MODE=0``, so the loop has to stay in
a ``torch.compiler.disable`` region; ``mark_step`` only splits a lazy graph.
A Python ``for`` left inside the compiled layer bakes the trip count into a new
graph per context length.

Query rows stay the padded prompt bucket. Context iteration follows the real
context length, so a short request does not run the long-context tiles.
"""

import os

import torch

_LENGTHS: tuple[int, int] | None = None


def enabled() -> bool:
    return os.environ.get("VLLM_HPU_PREFILL_TILE_REPLAY", "0").lower() in ("1", "true")


def context_tile() -> int:
    return int(os.environ.get("VLLM_HPU_PREFILL_CTX_TILE", "8192"))


def set_real_lengths(query_tokens: int, context_tokens: int) -> None:
    global _LENGTHS
    _LENGTHS = (int(query_tokens), int(context_tokens))


def clear_real_lengths() -> None:
    global _LENGTHS
    _LENGTHS = None


def real_lengths() -> tuple[int, int] | None:
    return _LENGTHS


def single_prompt_bucket(batch: int, query: int, block_size: int, max_model_len: int) -> tuple[int, int, int]:
    """One prompt shape: every request pads up to this bucket."""
    blocks = max(0, (max_model_len - query) // block_size)
    return (batch, query, blocks)


def context_tile_count(real_context: int, tile: int) -> int:
    if real_context <= 0:
        return 0
    return (real_context + tile - 1) // tile


# Prefill kernel batch. Counts 1..6 stay exact so a 49k prompt has no empty tiles.
# 32 * 8192 covers the 262144-token limit.
PREFILL_BUNDLE_COUNTS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
# Decode stays on the coarser seven. This prompt's decode already lands on 65536 tokens.
DECODE_BUNDLE_COUNTS = (1, 2, 4, 8, 16, 24, 32)


def bundle_count(n_tiles: int) -> int:
    """Smallest prefill bundle that can hold ``n_tiles``."""
    if n_tiles <= 0:
        return 0
    for size in PREFILL_BUNDLE_COUNTS:
        if n_tiles <= size:
            return size
    return PREFILL_BUNDLE_COUNTS[-1]


def prompt_bundle_buckets(batch: int, query: int, block_size: int, max_model_len: int) -> list[tuple[int, int, int]]:
    """Eleven prompt shapes, one per prefill bundle, capped at the model length."""
    per_tile = context_tile() // block_size
    max_blocks = max(0, (max_model_len - query) // block_size)
    buckets = []
    for count in PREFILL_BUNDLE_COUNTS:
        blocks = min(max_blocks, count * per_tile)
        bucket = (batch, query, blocks)
        if not buckets or buckets[-1] != bucket:
            buckets.append(bucket)
    return buckets


def decode_bundle_buckets(block_size: int, max_model_len: int) -> list[tuple[int, int, int]]:
    """Seven decode shapes, batch 1, query 1."""
    max_blocks = max(1, max_model_len // block_size)
    buckets = []
    for count in DECODE_BUNDLE_COUNTS:
        blocks = min(max_blocks, (count * context_tile()) // block_size)
        bucket = (1, 1, blocks)
        if not buckets or buckets[-1] != bucket:
            buckets.append(bucket)
    return buckets


def pad_block_window(block_list: torch.Tensor, start: int, n_blocks: int) -> torch.Tensor:
    """Fixed-length block-id window. The last context tile is shorter than ``n_blocks``."""
    remaining = int(block_list.numel()) - start
    if start >= 0 and remaining >= n_blocks:
        return block_list.narrow(0, start, n_blocks)
    if remaining <= 0:
        return block_list.new_zeros((n_blocks,))
    pad = block_list.new_zeros((n_blocks - remaining,))
    return torch.cat((block_list.narrow(0, start, remaining), pad))


def _count_tensor(value, ref: torch.Tensor) -> torch.Tensor:
    """A length as a tensor input. A Python int written on device becomes a compiled constant."""
    if isinstance(value, torch.Tensor):
        count = value.detach()
    else:
        count = torch.tensor(int(value), dtype=torch.int32)
    if count.device != ref.device:
        count = count.to(device=ref.device)
    if count.dtype != ref.dtype:
        count = count.to(dtype=ref.dtype)
    return count


def _column_mask(q_len: int, k_len: int, valid_k: int, *, causal: bool, real_q: int, device, dtype) -> torch.Tensor:
    """Additive bias. Invalid keys are -inf. The mask shape does not depend on the counts.

    ``dtype`` must match q/k/v. ``sdpa_recomp_fwd`` rejects a higher-precision bias
    (fp32 bias against bf16 activations fails graph compile, synStatus 26).
    """
    cols = torch.arange(k_len, device=device)
    valid = _count_tensor(valid_k, cols)
    mask = cols.view(1, 1, 1, k_len) >= valid
    if causal:
        rows = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
        real = _count_tensor(real_q, cols)
        mask = mask | (cols.view(1, 1, 1, k_len) > rows) | (cols.view(1, 1, 1, k_len) >= real)
    bias = torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)
    return bias.masked_fill(mask, float("-inf"))


def _bundle_mask(q_len: int, k_len: int, valid_ks: list[int], *, device, dtype) -> torch.Tensor:
    """Per-tile additive bias. Shape ``[n, 1, q_len, k_len]`` is the compiled bundle size."""
    n = len(valid_ks)
    cols = torch.arange(k_len, device=device)
    valid = torch.tensor(valid_ks, device=device, dtype=cols.dtype).view(n, 1, 1, 1)
    mask = cols.view(1, 1, 1, k_len) >= valid
    bias = torch.zeros((n, 1, q_len, k_len), device=device, dtype=dtype)
    return bias.masked_fill(mask, float("-inf"))


def _merge(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    outs = [out.float() for out, _, _ in parts]
    ms = [m.float() for _, m, _ in parts]
    ls = [l.float().clamp_min(1e-20) for _, _, l in parts]
    m_max = ms[0]
    for m in ms[1:]:
        m_max = torch.maximum(m_max, m)
    num = None
    den = None
    for out, m, l in zip(outs, ms, ls):
        weight = torch.exp(m - m_max) / l
        term = weight * out
        num = term if num is None else num + term
        den = weight if den is None else den + weight
    return num / den


def run_tiled_attention(query: torch.Tensor, n_ctx_tiles: int, load_ctx_tile, key_q: torch.Tensor,
                        value_q: torch.Tensor, real_q: int, sdpa, break_graph, scale: float,
                        softmax_mode: str) -> torch.Tensor:
    """Replay fixed-shape tiles. ``load_ctx_tile(i)`` returns ``(k, v, valid_k)`` with k/v already padded to the tile.

    ``query`` is ``[1, Hq, Q, D]``. Keys are ``[1, Hkv, K, D]``. ``sdpa`` returns ``(out, row_max, normalizer)``.
    """
    parts = []
    for tile_index in range(n_ctx_tiles):
        break_graph()
        key, value, valid_k = load_ctx_tile(tile_index)
        bias = _column_mask(query.shape[-2],
                            key.shape[-2],
                            valid_k,
                            causal=False,
                            real_q=real_q,
                            device=query.device,
                            dtype=query.dtype)
        parts.append(sdpa(query, key, value, bias, scale, softmax_mode))
    break_graph()
    bias = _column_mask(query.shape[-2],
                        key_q.shape[-2],
                        key_q.shape[-2],
                        causal=True,
                        real_q=real_q,
                        device=query.device,
                        dtype=query.dtype)
    parts.append(sdpa(query, key_q, value_q, bias, scale, softmax_mode))
    return _merge(parts).to(dtype=query.dtype)


def run_bundled_attention(query: torch.Tensor, n_ctx_tiles: int, load_ctx_tile, key_q: torch.Tensor,
                          value_q: torch.Tensor, real_q: int, sdpa, break_graph, scale: float,
                          softmax_mode: str) -> torch.Tensor:
    """One context kernel for the whole step. The batch is a bundle size, not the raw tile count.

    Padding slots past ``n_ctx_tiles`` stay in the kernel so the shape is stable, and are dropped
    before the merge. An all-masked slot makes ``sdpa_recomp_fwd`` return NaN, so those slots are
    given a finite mask and then discarded.
    """
    n_real = n_ctx_tiles
    n = bundle_count(n_real)
    parts = []
    if n:
        break_graph()
        keys = []
        values = []
        valids = []
        k_len = None
        for index in range(n):
            key, value, valid = load_ctx_tile(index)
            k_len = key.shape[-2]
            if index >= n_real:
                valid = k_len
            keys.append(key)
            values.append(value)
            valids.append(int(valid))
        query_b = query.expand(n, -1, -1, -1).contiguous()
        bias = _bundle_mask(query.shape[-2], k_len, valids, device=query.device, dtype=query.dtype)
        out, row_max, normalizer = sdpa(query_b, torch.cat(keys, dim=0), torch.cat(values, dim=0), bias, scale,
                                        softmax_mode)
        for index in range(n_real):
            parts.append((out[index], row_max[index], normalizer[index]))
    break_graph()
    bias = _column_mask(query.shape[-2],
                        key_q.shape[-2],
                        key_q.shape[-2],
                        causal=True,
                        real_q=real_q,
                        device=query.device,
                        dtype=query.dtype)
    parts.append(sdpa(query, key_q, value_q, bias, scale, softmax_mode))
    return _merge(parts).to(dtype=query.dtype)
