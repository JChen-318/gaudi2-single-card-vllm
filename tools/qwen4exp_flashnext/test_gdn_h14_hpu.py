# SPDX-License-Identifier: Apache-2.0
"""H26 regression gates for the Qwen4Exp GDN fast path (real Layer-0 weights).

Two contracts, both measured against the exact torch reference on the HPU:

**Part 1 -- prefill conv state.**
A bucketed prefill is padded to the bucket length while ``query_start_loc_p``
keeps the *actual* query lengths (``hpu_model_runner`` builds it from
``cumsum(query_lens)``). ``hpu_causal_conv1d_fn`` therefore gathers the new
conv state at ``actual_qlens + [0, 1, 2]`` inside ``cat(init_state, x)`` --
the last three *real* tokens -- not at the padded end of the slot. The fused
``gdn_conv_prefill`` TPC kernel used to publish ``next_state`` at
``stop == tokens_per_seq``, i.e. from the zeroed tail, so ``conv_state`` was
overwritten with ``[0, 0, 0]`` and the first decode token after every padded
prefill convolved against a wiped history. The gate is *bit identity*
(``conv_absmax == 0.0``) for B = 1..4, ``actual_qlen`` in
``{0, 1, 2, 3, 380, 512, 539, 911, 1536}``, homogeneous and heterogeneous
batches.

**Part 2 -- padded BF16 chunk recurrence.**
The chunk recurrence must be padding invariant (the padded tail is neutralized
through the gating mask, so the real rows must match an unpadded run) and BF16
compute must stay inside the layer-test tolerance against the FP32 production
configuration (``compute_dtype=torch.float32``, FlashQLA reformulation).

Run (both execution modes must pass):

  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_gdn_h14_hpu.py
  PT_HPU_LAZY_MODE=1 python tools/qwen4exp_flashnext/test_gdn_h14_hpu.py
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

import vllm_gaudi  # noqa: F401,E402  (registers the HPU platform first)

from test_gdn_layer_hpu import _build  # noqa: E402  (same real-weight fixture)

HIDDEN = 2560
# (name, per-sequence bucket length, actual unpadded lengths per sequence)
CONV_CASES = [
    ("B1_T64_real64", 64, [64]),
    ("B1_T512_real512", 512, [512]),
    ("B1_T512_real380", 512, [380]),
    ("B1_T768_real539", 768, [539]),
    ("B1_T1280_real911", 1280, [911]),
    ("B1_T2048_real1536", 2048, [1536]),
    ("B1_T512_real0", 512, [0]),
    ("B1_T512_real1", 512, [1]),
    ("B1_T512_real2", 512, [2]),
    ("B1_T512_real3", 512, [3]),
    ("B2_T512_real512", 512, [512, 512]),
    ("B2_T512_real384", 512, [384, 384]),
    ("B2_T512_het", 512, [380, 512]),
    ("B2_T512_het_edge", 512, [0, 1]),
    ("B2_T2048_het", 2048, [911, 1536]),
    ("B4_T512_het", 512, [380, 511, 0, 33]),
]

CHUNK_CASES = [(64, 64), (512, 380), (1280, 911), (2048, 1536), (2048, 1960),
               (2048, 2048)]


def _conv_state_case(layer, seq_len: int, lengths: list[int]):
    """Prefill conv state: fused kernel vs ``hpu_causal_conv1d_fn`` (bit exact)."""
    from vllm_gaudi.models.qwen4exp import gdn_h14
    from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn

    num_seqs = len(lengths)
    tokens = seq_len * num_seqs
    gen = torch.Generator(device="hpu").manual_seed(7)
    hidden = (torch.randn(tokens, HIDDEN, generator=gen, device="hpu") * 0.3).bfloat16()
    qkvz_w = layer.in_proj_qkvz.weight.detach()
    ba_w = layer.in_proj_ba.weight.detach()
    merged_w = torch.cat([qkvz_w, ba_w], dim=0).contiguous()
    conv_shape = layer.get_state_shape()[0]
    conv_dtype = layer.get_state_dtype()[0]
    qsl = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="hpu")
    idx = torch.arange(num_seqs, dtype=torch.int32, device="hpu")
    his = torch.ones(num_seqs, dtype=torch.int32, device="hpu")
    conv_w = layer.conv1d.weight.detach().view(layer.conv1d.weight.size(0),
                                               layer.conv1d.weight.size(2))
    mask2d = torch.zeros((num_seqs, seq_len), dtype=torch.bfloat16, device="hpu")
    for row, length in enumerate(lengths):
        mask2d[row, :length] = 1.0
    mask = mask2d.view(-1, 1)
    with torch.no_grad():
        mixed_qkv = torch.nn.functional.linear(hidden, merged_w)[:, :qkvz_w.shape[0]]
        mixed_qkv, _z = mixed_qkv.split([layer.qkv_size, layer.z_size], dim=-1)
        mixed_qkv = mixed_qkv * mask
        # identical non-zero history on both sides, so a wipe is visible
        seed = torch.arange((num_seqs + 2) * conv_shape[0] * conv_shape[1],
                            dtype=torch.float32, device="hpu").reshape(num_seqs + 2, *conv_shape)
        base = ((seed * 0.0625).sin() * 0.5).to(conv_dtype)
        pool_ref = base.clone()
        pool_h14 = base.clone()
        ref_out = hpu_causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1), weight=conv_w, bias=layer.conv1d.bias,
            activation=layer.activation, conv_states=pool_ref, has_initial_state=his,
            cache_indices=idx, query_start_loc=qsl, block_size_to_align=0,
            is_prompt=True).transpose(0, 1)
        got = gdn_h14.gdn_conv_prefill(mixed_qkv, pool_h14, gdn_h14.conv_weight_t(layer), idx,
                                       his, state_stops=qsl)
        assert got is not None, "gdn_conv_prefill declined the live prefill shape"
        h14_out, next_state = got
        safe = torch.remainder(idx.reshape(-1).to(torch.long), pool_h14.shape[0])
        pool_h14.index_copy_(0, safe, next_state)
        state_ref = pool_ref.index_select(0, safe)
        state_h14 = pool_h14.index_select(0, safe)
    return {
        "out_absmax": float((h14_out.float() - ref_out.float()).abs().max()),
        "state_absmax": float((state_h14.float() - state_ref.float()).abs().max()),
        "state_absmax_value": float(state_ref.abs().max()),
        "per_seq_state_absmax": [
            float((state_h14[s].float() - state_ref[s].float()).abs().max()) for s in range(num_seqs)
        ],
        "next_state": next_state,
        "mixed_qkv": mixed_qkv,
        "pool_h14": pool_h14,
        "pool_ref": pool_ref,
        "idx": idx,
        "qsl": qsl,
    }


def check_conv_state(layer) -> None:
    """H26 Part 1: the prefill conv state is bit-identical to the torch conv."""
    print("  -- Part 1: conv state at the last REAL token (padded spans)")
    for name, seq_len, lengths in CONV_CASES:
        lengths = [min(r, seq_len) for r in lengths]
        row = _conv_state_case(layer, seq_len, lengths)
        print(f"    {name:20s} real={lengths} state_absmax {row['state_absmax']:.3e} "
              f"out_absmax {row['out_absmax']:.3e} |state| {row['state_absmax_value']:.3e}")
        assert row["state_absmax"] == 0.0, (
            f"{name}: conv state differs from hpu_causal_conv1d_fn by "
            f"{row['state_absmax']} (per sequence {row['per_seq_state_absmax']})")
        # a real span with has_initial must always publish a non-zero history
        if max(lengths) > 0:
            assert row["state_absmax_value"] > 0.0, f"{name}: conv state is all zero"
        # one bf16 ULP: the TPC kernel sums the 4 taps in F32, the torch
        # reference accumulates tap by tap in the cache dtype.
        assert row["out_absmax"] < 5e-3, f"{name}: conv output absmax {row['out_absmax']}"


def check_decode_after_padded_prefill(layer) -> None:
    """H26 Part 1: the first decode token after a padded prefill must match.

    This is the live failure mode: with the wiped state the decode conv saw
    ``[0, 0, 0]`` instead of the last three prompt tokens. The reference is the
    direct tap-ordered formula over the same BF16 state row (``silu`` of the
    four taps in FP32), which is the contract ``_h14_gdn_conv_decode``
    implements; ``hpu_causal_conv1d_update`` is not used because its 2-D input
    path does not reshape to the 3-D buffer its inner kernel unpacks (a
    pre-existing bug on the *fallback* decode conv, outside this unit).
    """
    from vllm_gaudi.models.qwen4exp import gdn_h14

    print("  -- Part 1: first decode step after a padded prefill")
    seq_len, real, num_seqs = 1280, 911, 1
    row = _conv_state_case(layer, seq_len, [real] * num_seqs)
    gen = torch.Generator(device="hpu").manual_seed(11)
    token = (torch.randn(num_seqs, layer.qkv_size, generator=gen, device="hpu") * 0.3).bfloat16()
    weight_t = gdn_h14.conv_weight_t(layer)  # [4, C] tap-major
    # this checkpoint has no conv bias (the TPC conv kernels take none)
    bias = (layer.conv1d.bias.detach().float() if layer.conv1d.bias is not None
            else torch.zeros(layer.qkv_size, dtype=torch.float32, device="hpu"))
    idx = row["idx"]
    safe = torch.remainder(idx.reshape(-1).to(torch.long), row["pool_h14"].shape[0])
    with torch.no_grad():
        state_before = row["pool_h14"].index_select(0, safe).clone()
        dec_h14 = gdn_h14.gdn_conv_decode(token, row["pool_h14"],
                                          weight_t, idx)
        assert dec_h14 is not None, "gdn_conv_decode declined the live decode shape"
        dec_out, dec_state = dec_h14
        row["pool_h14"].index_copy_(0, safe, dec_state)
        dec_state = row["pool_h14"].index_select(0, safe)
        # tap-ordered reference over the *same* (post-prefill) BF16 history
        taps = torch.cat((state_before, token.reshape(num_seqs, 1, -1)), dim=1)  # [B, 4, C]
        ref_out = torch.nn.functional.silu(
            (taps.float() * weight_t.float().unsqueeze(0)).sum(1) + bias).bfloat16()
        ref_state = taps[:, 1:, :]
    out_absmax = float((dec_out.float() - ref_out.float()).abs().max())
    state_absmax = float((dec_state.float() - ref_state.float()).abs().max())
    wiped_absmax = float((torch.nn.functional.silu(
        (torch.cat((torch.zeros_like(state_before), token.reshape(num_seqs, 1, -1)),
                   dim=1).float() * weight_t.float().unsqueeze(0)).sum(1) + bias).bfloat16().float()
        - ref_out.float()).abs().max())
    print(f"    decode after real={real} in T={seq_len}: out_absmax {out_absmax:.3e} "
          f"state_absmax {state_absmax:.3e} | the pre-H26 wiped state was off by "
          f"{wiped_absmax:.3e}")
    assert out_absmax < 5e-3, f"first decode token differs by {out_absmax}"
    assert state_absmax == 0.0, f"decode state differs by {state_absmax}"


def _chunk_inputs(layer, tokens: int, real: int, hidden: torch.Tensor | None = None):
    """Layer-0 chunk recurrence inputs for a span of ``real`` tokens in ``tokens``."""
    from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_fn
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating

    if hidden is None:
        gen = torch.Generator(device="hpu").manual_seed(5)
        hidden = (torch.randn(tokens, HIDDEN, generator=gen, device="hpu") * 0.3).bfloat16()
    qkvz_w = layer.in_proj_qkvz.weight.detach()
    ba_w = layer.in_proj_ba.weight.detach()
    merged_w = torch.cat([qkvz_w, ba_w], dim=0).contiguous()
    conv_shape, ssm_shape = layer.get_state_shape()[:2]
    conv_dtype, ssm_dtype = layer.get_state_dtype()
    pool = torch.zeros((3, *conv_shape), dtype=conv_dtype, device="hpu")
    idx = torch.zeros(1, dtype=torch.int32, device="hpu")
    his = torch.zeros(1, dtype=torch.int32, device="hpu")
    qsl = torch.tensor([0, real], dtype=torch.int32, device="hpu")
    conv_w = layer.conv1d.weight.detach().view(layer.conv1d.weight.size(0),
                                               layer.conv1d.weight.size(2))
    mask = torch.zeros((tokens, 1), dtype=torch.bfloat16, device="hpu")
    mask[:real] = 1.0
    with torch.no_grad():
        merged = torch.nn.functional.linear(hidden, merged_w)
        mixed_qkvz = merged[:, :qkvz_w.shape[0]]
        ba = merged[:, qkvz_w.shape[0]:]
        mixed_qkv, _z = mixed_qkvz.split([layer.qkv_size, layer.z_size], dim=-1)
        b, a = ba.chunk(2, dim=-1)
        b, a = b.contiguous(), a.contiguous()
        mixed_qkv = mixed_qkv * mask
        g, beta = hpu_fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        keep = mask.view(1, -1, 1).to(g.dtype)
        g = g * keep
        beta = beta * keep
        conv = hpu_causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1), weight=conv_w, bias=layer.conv1d.bias,
            activation=layer.activation, conv_states=pool, has_initial_state=his,
            cache_indices=idx, query_start_loc=qsl, block_size_to_align=0,
            is_prompt=True).transpose(0, 1)
        conv = conv * mask
        q, k, v = layer.rearrange_mixed_qkv(conv.contiguous())
        init_state = torch.zeros(1, *ssm_shape, dtype=ssm_dtype, device="hpu")
    return dict(q=q, k=k, v=v, g=g, beta=beta, init_state=init_state, seq_len=tokens)


def _chunk_run(layer, d, **extra):
    from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_chunk_gated_delta_rule

    kwargs = dict(q=d["q"], k=d["k"], v=d["v"], g=d["g"], beta=d["beta"],
                  initial_state=d["init_state"], output_final_state=True,
                  use_qk_l2norm_in_kernel=True, chunk_size=layer.mamba_chunk_size,
                  prefill_num_seqs=1, prefill_seq_len=d["seq_len"],
                  neumann_iters=layer.gdn_neumann_iters,
                  qk_head_repeat_override=layer.gdn_native_qk_head_repeat,
                  flashqla_reformulation=True)
    kwargs.update(extra)
    return hpu_chunk_gated_delta_rule(**kwargs)


def check_chunk_recurrence(layer) -> None:
    """H26 Part 2: the shipped BF16 chunk config on padded spans.

    Shipped configuration: BF16 MME bulk products, the exact FP32 unit-lower
    solve (``solve_in_fp32`` + ``recursive_solver_base=2``) and the recurrent
    state in the BF16 cache dtype. The reference is the FP32 production chunk
    recurrence (Neumann solve, ``compute_dtype=torch.float32``).
    """
    print("  -- Part 2: padded chunk recurrence, shipped BF16 config vs the FP32 path")
    shipped = dict(compute_dtype=torch.bfloat16, solve_in_fp32=True, recursive_solver_base=2)
    for tokens, real in CHUNK_CASES:
        d = _chunk_inputs(layer, tokens, real)
        with torch.no_grad():
            ref_out, ref_state = _chunk_run(layer, d, compute_dtype=torch.float32)
            out, state = _chunk_run(layer, d, **shipped)
        ref_out = ref_out.float()
        out_f = out.float()
        diff = out_f[0, :real] - ref_out[0, :real]
        rel = float(diff.norm() / ref_out[0, :real].norm().clamp(min=1e-12))
        absmax = float(diff.abs().max())
        cos = float(torch.nn.functional.cosine_similarity(out_f[0, :real].reshape(1, -1),
                                                          ref_out[0, :real].reshape(1, -1)))
        state_absmax = float((state.float() - ref_state.float()).abs().max())
        pad_absmax = float(out_f[0, real:].abs().max()) if real < tokens else 0.0
        print(f"    T={tokens:5d} real={real:5d} absmax {absmax:.3e} rel_l2 {rel:.3e} "
              f"cos {cos:.6f} state {state_absmax:.3e} pad {pad_absmax:.3e}")
        assert torch.isfinite(out_f).all(), (tokens, real)
        assert rel < 2e-2, f"T={tokens} real={real}: BF16 rel_l2 {rel}"
        assert absmax < 5e-3, f"T={tokens} real={real}: BF16 absmax {absmax}"
        assert cos > 0.999, f"T={tokens} real={real}: cosine {cos}"
        assert state_absmax < 2e-2, f"T={tokens} real={real}: state {state_absmax}"
        # the gating mask must leave the padded tail exactly neutral
        assert pad_absmax == 0.0, f"T={tokens} real={real}: padded rows {pad_absmax}"


def check_padding_invariance(layer) -> None:
    """H26 Part 2: the same real tokens give the same rows padded or not."""
    print("  -- Part 2: padding invariance (911 in 1280 is the H25 live failure)")
    for real, bucket in ((911, 1280), (1536, 2048), (380, 512)):
        gen = torch.Generator(device="hpu").manual_seed(5)
        hidden = (torch.randn(bucket, HIDDEN, generator=gen, device="hpu") * 0.3).bfloat16()
        plain = _chunk_inputs(layer, real, real, hidden=hidden[:real].contiguous())
        padded = _chunk_inputs(layer, bucket, real, hidden=hidden)
        for label, spec in (("fp32", dict(compute_dtype=torch.float32)),
                            ("bf16", dict(compute_dtype=torch.bfloat16))):
            with torch.no_grad():
                out_plain = _chunk_run(layer, plain, **spec)[0].float()
                out_padded = _chunk_run(layer, padded, **spec)[0].float()
            diff = out_padded[0, :real] - out_plain[0, :real]
            rel = float(diff.norm() / out_plain[0, :real].norm().clamp(min=1e-12))
            print(f"    real={real:5d} bucket={bucket:5d} {label:4s} absmax "
                  f"{float(diff.abs().max()):.3e} rel_l2 {rel:.3e}")
            assert rel < 2e-2, f"real={real} bucket={bucket} {label}: rel_l2 {rel}"


def main() -> int:
    print(f"H26 GDN gates (PT_HPU_LAZY_MODE={os.environ.get('PT_HPU_LAZY_MODE')})")
    layer = _build().to("hpu")
    layer.cache_group_idx = torch.tensor([0], dtype=torch.long, device="hpu")
    check_conv_state(layer)
    check_decode_after_padded_prefill(layer)
    check_chunk_recurrence(layer)
    check_padding_invariance(layer)
    torch.hpu.synchronize()
    print("all H26 GDN tests passed")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
    raise SystemExit(main())
