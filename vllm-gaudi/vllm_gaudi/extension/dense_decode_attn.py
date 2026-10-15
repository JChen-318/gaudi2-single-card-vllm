# SPDX-License-Identifier: Apache-2.0
"""Shared-prefix decode attention over a request's whole context at once.

``flat_pa_query_block`` works page by page: the query block is copied to every
page, each page keeps its own softmax max/sum and its own [heads, width,
head_size] partial output, and the partial outputs are rescaled and summed per
request at the end. All of that is proportional to the number of pages. With an
FP8 KV cache the pages are also gathered, widened to BF16 and narrowed back to
FP8 for every matmul, and the softmax probabilities and partial outputs are
squeezed into FP8 with a unit scale.

Here each request's pages are gathered once (kept in the cache dtype) and the
rows of all requests attend over the concatenated keys with one masked softmax:
a row only sees the pages of its own request (``block_groups``) and, inside
them, what ``block_bias`` allows. Nothing is materialised per page beyond the
gathered K/V and the [kv_heads, rows, keys] scores. No HPU import at module
level, so the math can be checked on CPU.
"""
from __future__ import annotations

import torch

# Probabilities are <= 1; scale them into the E4M3 range before an FP8 AV matmul
# (Gaudi2's E4M3 tops out at 240) so that small weights are not flushed to zero.
PROB_SCALE = 128.0
FP8 = torch.float8_e4m3fn


def _on_hpu(t: torch.Tensor) -> bool:
    return t.device.type == "hpu"


def to_fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    if _on_hpu(x):
        return torch.ops.hpu.cast_to_fp8_v2(x, scale, False, False, FP8)[0]
    return (x.float() * scale).to(FP8)


def from_fp8(x: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if _on_hpu(x):
        return torch.ops.hpu.cast_from_fp8(x, 1.0, dtype)
    return x.to(dtype)


def fp8_bmm(a: torch.Tensor, b: torch.Tensor, trans_b: bool, a_scale_inv: float = 1.0) -> torch.Tensor:
    """Batched FP8 x FP8 matmul with BF16 output; ``a`` was scaled by 1 / a_scale_inv."""
    if _on_hpu(a):
        return torch.ops.hpu.fp8_gemm_v2(A=a, trans_A=False, B=b, trans_B=trans_b, D=None, out_dtype=torch.bfloat16,
                                         A_scale_inv=a_scale_inv, B_scale_inv=1.0, bias=None, accumulate=False)
    bf = b.float().transpose(-2, -1) if trans_b else b.float()
    return torch.matmul(a.float() * a_scale_inv, bf).to(torch.bfloat16)


def dense_query_block_attention(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                                block_list: torch.Tensor, block_bias: torch.Tensor, block_groups: torch.Tensor,
                                block_size: int, scale: float, qk: str = "fp8", av: str = "fp8") -> torch.Tensor:
    """``query`` [requests, width, q_heads * head]; caches [slots, kv_heads, head]; ``block_list`` [pages];
    ``block_bias`` [pages, width, block_size] (0 / -inf); ``block_groups`` [pages] (request row, -1 = padding).
    ``qk`` / ``av``: "fp8" or "bf16" matmul for an FP8 cache (a BF16 cache always uses BF16).
    Returns [requests * width, q_heads, head]."""
    requests, width, hidden = query.shape
    _, kv_heads, head = key_cache.shape
    q_heads = hidden // head
    group = q_heads // kv_heads
    pages = block_list.shape[0]
    keys = pages * block_size
    fp8_cache = key_cache.dtype == FP8

    def gather(cache: torch.Tensor) -> torch.Tensor:  # [kv_heads, keys, head], cache dtype
        return cache.unflatten(0, (-1, block_size)).index_select(0, block_list).flatten(0, 1).transpose(0, 1)

    k = gather(key_cache)
    v = gather(value_cache)
    # rows ordered (request, token, head within the kv group) under each kv head
    q = (query * scale).view(requests, width, kv_heads, group, head).permute(2, 0, 1, 3, 4)
    q = q.reshape(kv_heads, requests * width * group, head)
    if fp8_cache and qk == "fp8":
        scores = fp8_bmm(to_fp8(q, 1.0), k, trans_b=True)
    else:
        kk = from_fp8(k, query.dtype) if fp8_cache else k
        scores = torch.matmul(q, kk.transpose(-2, -1))

    # row (request r, token t) sees slot (page p, s) iff p belongs to r and block_bias[p, t, s] == 0
    bias = block_bias.to(scores.dtype).permute(1, 0, 2).reshape(1, width, keys)
    owner = block_groups.view(1, pages, 1) == torch.arange(requests, device=block_groups.device).view(requests, 1, 1)
    owner = owner.expand(requests, pages, block_size).reshape(requests, 1, keys)
    mask = torch.where(owner, bias, torch.full_like(bias, float("-inf")))  # [requests, width, keys]
    scores = scores.view(kv_heads, requests, width, group, keys) + mask.view(1, requests, width, 1, keys)
    # padding rows see nothing: keep their max finite and their sum non-zero (output 0, no NaN)
    row_max = scores.amax(dim=-1, keepdim=True).clamp(min=torch.finfo(scores.dtype).min)
    probs = (scores - row_max).exp()
    row_sum = probs.sum(dim=-1, keepdim=True, dtype=torch.float32).clamp(min=1e-30)
    probs = probs.view(kv_heads, requests * width * group, keys)
    if fp8_cache and av == "fp8":
        out = fp8_bmm(to_fp8(probs, PROB_SCALE), v, trans_b=False, a_scale_inv=1.0 / PROB_SCALE)
    else:
        vv = from_fp8(v, query.dtype) if fp8_cache else v
        out = torch.matmul(probs.to(vv.dtype), vv)
    out = out.view(kv_heads, requests, width, group, head) / row_sum.to(out.dtype)
    return out.permute(1, 2, 0, 3, 4).reshape(requests * width, q_heads, head).to(query.dtype)
