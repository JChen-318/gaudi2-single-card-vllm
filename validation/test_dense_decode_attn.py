"""Dense shared-prefix decode attention (VLLM_HPU_SHARED_PREFIX_ATTN=dense) against a float64 reference.

Page lists, groups and per-token visibility come from decode_shared_prefix.shared_prefix_buffers, the bias from
the runner's rule (slot >= usage -> -inf), so the inputs have the production layout: scattered physical pages,
padding pages (group -1) and verification blocks that cross a page boundary."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import v6_support  # noqa: E402

dense = v6_support._load("extension/dense_decode_attn.py", "dense_decode_attn")
shared = v6_support._load("v1/worker/decode_shared_prefix.py", "decode_shared_prefix_dense")

BLOCK, KV, GROUP, HEAD, WIDTH = 16, 2, 3, 32, 8
FP8 = torch.float8_e4m3fn


def make(context_lens, seed, fp8=True, pad_to=None):
    gen = torch.Generator().manual_seed(seed)
    requests = len(context_lens)
    query_lens = [WIDTH] * requests
    pages_needed = [(c + WIDTH + BLOCK - 1) // BLOCK for c in context_lens]
    total_pages = 200
    perm = torch.randperm(total_pages - 1, generator=gen).numpy()  # the last page is the padding page
    table = np.zeros((requests, max(pages_needed)), dtype=np.int64)
    used = 0
    for r, n in enumerate(pages_needed):
        table[r, :n] = perm[used:used + n]
        used += n
    bucket = (lambda n: pad_to) if pad_to else (lambda n: n)
    block_list, groups, usage = shared.shared_prefix_buffers(context_lens, query_lens, WIDTH, BLOCK, table, total_pages - 1, bucket,
                                                            torch.int64)
    mask = torch.arange(BLOCK).unsqueeze(0) >= usage.unsqueeze(-1)
    bias = torch.zeros(mask.shape, dtype=torch.bfloat16).masked_fill_(mask, -math.inf)
    cache_dtype = FP8 if fp8 else torch.bfloat16
    k_cache = (torch.randn(total_pages * BLOCK, KV, HEAD, generator=gen) * 0.8).to(cache_dtype)
    v_cache = (torch.randn(total_pages * BLOCK, KV, HEAD, generator=gen) * 0.8).to(cache_dtype)
    query = torch.randn(requests, WIDTH, KV * GROUP * HEAD, generator=gen).to(torch.bfloat16)
    return query, k_cache, v_cache, block_list, bias, groups, table, pages_needed


def reference(query, k_cache, v_cache, table, pages_needed, context_lens, scale):
    """Causal attention per request in float64 over the request's own pages."""
    requests = query.shape[0]
    out = torch.zeros(requests, WIDTH, KV * GROUP, HEAD, dtype=torch.float64)
    q = query.double().view(requests, WIDTH, KV * GROUP, HEAD) * scale
    for r in range(requests):
        slots = torch.cat([torch.arange(p * BLOCK, (p + 1) * BLOCK) for p in table[r, :pages_needed[r]]])
        k = k_cache[slots].double()
        v = v_cache[slots].double()
        for t in range(WIDTH):
            visible = context_lens[r] + t + 1
            for h in range(KV * GROUP):
                s = k[:visible, h // GROUP] @ q[r, t, h]
                w = torch.softmax(s, dim=0)
                out[r, t, h] = w @ v[:visible, h // GROUP]
    return out.view(requests * WIDTH, KV * GROUP, HEAD)


@pytest.mark.parametrize("context_lens", [[5], [40], [37, 90], [120, 3]])
@pytest.mark.parametrize("qk,av", [("bf16", "bf16"), ("fp8", "bf16"), ("bf16", "fp8"), ("fp8", "fp8")])
def test_matches_reference_fp8_cache(context_lens, qk, av):
    scale = HEAD**-0.5
    query, k, v, block_list, bias, groups, table, pages = make(context_lens, seed=len(context_lens) * 7 + 1,
                                                              pad_to=sum((c + WIDTH + BLOCK - 1) // BLOCK
                                                                         for c in context_lens) + 3)
    got = dense.dense_query_block_attention(query, k, v, block_list, bias, groups, BLOCK, scale, qk=qk, av=av).double()
    want = reference(query, k, v, table, pages, context_lens, scale)
    err = (got - want).abs().max().item()
    # BF16 matmuls: BF16 rounding only. FP8 QK quantises the query (3 mantissa bits); FP8 AV the probabilities.
    tol = {("bf16", "bf16"): 0.03, ("fp8", "bf16"): 0.12, ("bf16", "fp8"): 0.08, ("fp8", "fp8"): 0.15}[(qk, av)]
    assert err < tol, err
    assert torch.isfinite(got).all()


def test_bf16_cache_and_padding_request_rows():
    scale = HEAD**-0.5
    context_lens = [33]
    query, k, v, block_list, bias, groups, table, pages = make(context_lens, seed=3, fp8=False, pad_to=8)
    # a padded batch: a second request row that owns no page must come out finite (zeros)
    query2 = torch.cat([query, torch.randn_like(query)])
    got = dense.dense_query_block_attention(query2, k, v, block_list, bias, groups, BLOCK, scale).double()
    want = reference(query, k, v, table, pages, context_lens, scale)
    assert (got[:WIDTH] - want).abs().max().item() < 0.03
    assert torch.isfinite(got).all() and got[WIDTH:].abs().max().item() == 0


def test_rows_never_see_other_requests():
    scale = HEAD**-0.5
    context_lens = [50, 50]
    query, k, v, block_list, bias, groups, table, pages = make(context_lens, seed=11)
    base = dense.dense_query_block_attention(query, k, v, block_list, bias, groups, BLOCK, scale, "bf16", "bf16")
    # change only the second request's pages: the first request's rows must not move
    k2, v2 = k.clone(), v.clone()
    for p in table[1, :pages[1]]:
        k2[p * BLOCK:(p + 1) * BLOCK] = (torch.randn(BLOCK, KV, HEAD) * 3).to(k.dtype)
        v2[p * BLOCK:(p + 1) * BLOCK] = (torch.randn(BLOCK, KV, HEAD) * 3).to(v.dtype)
    moved = dense.dense_query_block_attention(query, k2, v2, block_list, bias, groups, BLOCK, scale, "bf16", "bf16")
    assert torch.equal(base[:WIDTH], moved[:WIDTH])
    assert not torch.equal(base[WIDTH:], moved[WIDTH:])
