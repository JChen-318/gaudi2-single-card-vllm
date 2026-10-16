"""Fine prefill context buckets (VLLM_HPU_PROMPT_FINE_CTX_STEP) and the split prefill attention
(VLLM_HPU_PREFILL_SPLIT_ATTN): bucket lists and lookups, the frozen per-query context index, eligibility, and the
batched-slice attention with a CPU stand-in for the FusedSDPA kernel against dense softmax attention.

    PYTHONPATH=vllm-gaudi python -m pytest -q validation/core/test_prefill_buckets_slicing.py
"""
import math
import sys
import types

import pytest
import torch

try:
    import habana_frameworks.torch  # noqa: F401
except ImportError:  # utils.py imports it at module level; nothing here touches the device
    for name in ("habana_frameworks", "habana_frameworks.torch"):
        sys.modules.setdefault(name, types.ModuleType(name))

common = pytest.importorskip("vllm_gaudi.extension.bucketing.common")

BLOCK = 128
QUERIES = [128, 256, 512, 1024, 2048]
CTX = [0, 1, 2, 4, 8, 16, 32, 64, 128, 255, 509, 1017, 2047]  # the measured exponential context buckets
BUCKETS = sorted((1, q, c) for q in QUERIES for c in CTX)


def test_fine_buckets_only_for_largest_query(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_PROMPT_FINE_CTX_STEP", "64")
    monkeypatch.setenv("VLLM_HPU_PROMPT_FINE_CTX_MIN", "256")
    merged = common.add_fine_ctx_buckets(BUCKETS)
    added = sorted(set(merged) - set(BUCKETS))
    assert added == [(1, 2048, c) for c in range(256, 2047, 64)]
    assert merged == sorted(merged)
    # a 2048-token chunk at 95K tokens of context (743 pages) now pads to 768 pages instead of 1017
    assert common.find_equal_or_closest_greater_config(merged, (1, 2048, 743)) == (1, 2048, 768)
    assert common.find_equal_or_closest_greater_config(merged, (1, 2040, 1407)) == (1, 2048, 1408)
    assert common.find_equal_or_closest_greater_config(merged, (1, 1500, 743)) == (1, 2048, 768)
    assert common.find_equal_or_closest_greater_config(merged, (1, 1000, 743)) == (1, 1024, 1017)
    assert common.find_equal_or_closest_greater_config(merged, (1, 2048, 2047)) == (1, 2048, 2047)


def test_context_index(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_PROMPT_FINE_CTX_STEP", "64")
    index = common.index_prompt_ctx(common.add_fine_ctx_buckets(BUCKETS), BLOCK)
    assert common.PROMPT_CTX_BLOCK_SIZE == BLOCK and index is common.PROMPT_CTX_BY_QUERY
    assert index[1024] == tuple(CTX) and 768 in index[2048] and index[2048] == tuple(sorted(index[2048]))


def test_fine_buckets_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_HPU_PROMPT_FINE_CTX_STEP", raising=False)
    assert common.add_fine_ctx_buckets(BUCKETS) == BUCKETS


def cpu_chunk_kernel(q, k, v, mask, dropout_p, scale, is_causal, softmax_mode):
    """Same contract as the HPU kernel wrapper: per-slice normalised output, row max and 1/sum (float32)."""
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if mask is not None:
        s = s + mask.float()
    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - m)
    total = p.sum(dim=-1, keepdim=True)
    return torch.matmul(p, v.float()) / total, m, 1.0 / total


def dense(q, k, v, bias, scale):
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale + bias.float()
    return torch.matmul(torch.softmax(s, dim=-1), v.float())


split = pytest.importorskip("vllm_gaudi.extension.prefill_split_attn")


def cpu_sdpa_gqa(q, k, v, mask, scale, softmax_mode):
    """CPU stand-in for sdpa_recomp_fwd on GQA-shaped inputs, same (output, max, 1/sum) contract."""
    out, m, linv = cpu_chunk_kernel(q, k, v, mask, 0.0, scale, False, softmax_mode)
    return out, m, linv * 0.25  # a constant factor in the normaliser must cancel


@pytest.mark.parametrize("ctx_real,prefix_slices", [(5 * 64 - 7, 5), (4 * 64 + 1, 5), (3 * 64, 3)])
def test_split_attention_matches_dense(ctx_real, prefix_slices):
    """Scaled down: 64-key slices, a 48-token chunk, GQA 6 query heads over 2 KV heads. Padding (large values)
    fills the context from ctx_real to the end of the last slice."""
    slice_size, q_len, kv_heads, group, dim = 64, 48, 2, 3, 16
    prefix = prefix_slices * slice_size
    g = torch.Generator().manual_seed(ctx_real)
    q = torch.randn(1, kv_heads * group, q_len, dim, generator=g)
    k = torch.randn(1, kv_heads, prefix + q_len, dim, generator=g)
    v = torch.randn(1, kv_heads, prefix + q_len, dim, generator=g)
    k[:, :, ctx_real:prefix] = 5.0
    v[:, :, ctx_real:prefix] = 50.0
    bias = torch.zeros(1, 1, q_len, prefix + q_len)
    bias[..., ctx_real:prefix] = -math.inf
    bias[..., prefix:] = torch.triu(torch.full((q_len, q_len), -math.inf), diagonal=1)
    scale = dim**-0.5
    got = split.split_attention(q, k, v, bias, scale, slice_size=slice_size, sdpa=cpu_sdpa_gqa)
    want = dense(q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1), bias, scale)
    assert got.shape == q.shape
    assert torch.allclose(got, want, atol=2e-5, rtol=1e-4)


def test_split_eligibility():
    q = torch.zeros(1, 24, 2048, 8)
    bias = torch.zeros(1)

    def ok(prefix, q_len=2048, causal=True, b=bias):
        return split.eligible(q[:, :, :q_len], torch.zeros(1, 4, prefix + q_len, 8), b, causal, 2048, 32768)

    assert ok(32768) and ok(8192 * 23)
    assert not ok(16384)  # below the fine bucket range
    assert not ok(262016)  # the last exponential bucket: padding may span many slices
    assert not ok(32768, q_len=1024) and not ok(32768, causal=False) and not ok(32768, b=None)
