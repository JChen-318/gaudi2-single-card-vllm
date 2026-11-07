# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the split-key path of vllm_gaudi/omni/attention.py: an unaligned key length runs as an aligned
head plus a short tail, merged exactly through the blocks' row maxima and reciprocal row sums."""
import importlib
import sys
import types

import pytest
import torch


def _load(monkeypatch):
    # vllm_omni is not needed for these checks: stub the two base classes the module derives from.
    abstract = types.ModuleType("vllm_omni.diffusion.attention.backends.abstract")
    abstract.AttentionMetadata = object
    sdpa = types.ModuleType("vllm_omni.diffusion.attention.backends.sdpa")

    class SDPAImpl:
        def __init__(self, causal=False, softmax_scale=None):
            self.causal = causal
            self.softmax_scale = softmax_scale

    sdpa.SDPABackend = object
    sdpa.SDPAImpl = SDPAImpl
    for name in ("vllm_omni", "vllm_omni.diffusion", "vllm_omni.diffusion.attention",
                 "vllm_omni.diffusion.attention.backends"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, abstract.__name__, abstract)
    monkeypatch.setitem(sys.modules, sdpa.__name__, sdpa)
    monkeypatch.delitem(sys.modules, "vllm_gaudi.omni.attention", raising=False)
    return importlib.import_module("vllm_gaudi.omni.attention")


def _reference(q, k, v, scale):
    s = (q.double() @ k.double().transpose(-1, -2)) * scale
    return torch.softmax(s, dim=-1) @ v.double()


def _fake_recomp(factor):
    """CPU stand-in for torch.ops.hpu.sdpa_recomp_fwd: (normalised output, row max, factor / row sum)."""

    def run(q, k, v, mask, dropout, scale, causal, recompute, mode, valid, side):
        assert mask is None and not causal
        assert k.is_contiguous() and v.is_contiguous()
        s = (q.float() @ k.float().transpose(-1, -2)) * scale
        m = s.amax(-1, keepdim=True)
        p = torch.exp(s - m)
        total = p.sum(-1, keepdim=True)
        return ((p @ v.float()) / total).to(q.dtype), m, factor / total, None

    return run


def test_split_point(monkeypatch):
    attn = _load(monkeypatch)
    assert attn.kv_split_point(37750, 128) == 37632
    assert attn.kv_split_point(37632, 128) == 0  # aligned: one call
    assert attn.kv_split_point(4000, 128) == 0  # short: one call
    assert attn.kv_split_point(37750, 0) == 0  # switched off


@pytest.mark.parametrize("factor", [1.0, 1 / 128])  # "None" mode and the constant-scaled "fast" normaliser
@pytest.mark.parametrize("q_len,kv_len,heads,kv_heads", [(300, 4100, 4, 4), (257, 5000, 4, 2)])
def test_split_forward_matches_full_attention(monkeypatch, factor, q_len, kv_len, heads, kv_heads):
    attn = _load(monkeypatch)
    monkeypatch.setattr(attn, "HPU_SDPA_KV_SPLIT_MIN", 1024)
    monkeypatch.setattr(attn, "kv_split_point", lambda n, align=128: n // align * align if n % align else 0)
    monkeypatch.setattr(attn, "_sdpa_with_stats",
                        lambda q, k, v, scale: _fake_recomp(factor)(q, k, v, None, 0.0, scale, False, True, "None",
                                                                     None, "right")[:3])
    torch.manual_seed(0)
    D = 64
    q = torch.randn(1, q_len, heads, D)  # [B, L, H, D] as the backend receives it
    k = torch.randn(1, kv_len, kv_heads, D) * 2
    v = torch.randn(1, kv_len, kv_heads, D)
    impl = attn.HPUSDPAImpl(causal=False, softmax_scale=None)
    split = kv_len // 128 * 128
    out = impl._split_kv_forward(q.permute(0, 2, 1, 3).contiguous(), k, v, split, kv_len)
    rep = heads // kv_heads
    ref = _reference(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3).repeat_interleave(rep, 1),
                     v.permute(0, 2, 1, 3).repeat_interleave(rep, 1), D**-0.5).permute(0, 2, 1, 3)
    assert out.shape == (1, q_len, heads, D)
    assert torch.allclose(out.double(), ref, atol=2e-5, rtol=1e-4)


def test_merge_in_bf16_stays_close(monkeypatch):
    """The output-dtype update o1 + (o2 - o1) * share keeps BF16 accuracy when the tail block is short."""
    attn = _load(monkeypatch)
    torch.manual_seed(1)
    q = torch.randn(1, 2, 128, 64).bfloat16()
    k = torch.randn(1, 2, 4096 + 100, 64).bfloat16()
    v = torch.randn(1, 2, 4096 + 100, 64).bfloat16()
    run = _fake_recomp(1.0)
    o1, m1, l1, _ = run(q, k[:, :, :4096].contiguous(), v[:, :, :4096].contiguous(), None, 0, 0.125, False, True,
                        "None", None, "right")
    o2, m2, l2, _ = run(q, k[:, :, 4096:].contiguous(), v[:, :, 4096:].contiguous(), None, 0, 0.125, False, True,
                        "None", None, "right")
    out = attn.merge_kv_split(o1, m1, l1, o2, m2, l2)
    ref = _reference(q, k, v, 0.125)
    rel = ((out.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert out.dtype == torch.bfloat16
    assert rel < 1e-2


def test_boolean_mask_becomes_equivalent_additive_bias(monkeypatch):
    """A boolean key mask is passed to FusedSDPA as a 0/-inf bias (the boolean form is slow for some lengths)."""
    attn = _load(monkeypatch)
    torch.manual_seed(2)
    mask = torch.rand(1, 1, 5, 9) > 0.3
    mask[..., 0] = True  # every row keeps a key
    bias = attn.additive_mask(mask, torch.bfloat16)
    assert bias.dtype == torch.bfloat16 and bias.shape == mask.shape
    assert torch.equal(bias == 0, mask) and torch.isinf(bias[~mask]).all()
    q, k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 9, 8), torch.randn(1, 2, 9, 8)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.float())
    assert torch.allclose(out, ref, atol=1e-6)
    float_mask = torch.zeros(1, 1, 5, 9)
    assert attn.additive_mask(float_mask, torch.bfloat16) is float_mask
