"""HPU benchmark: chunked-prefill attention (one 2048-token chunk against a long cached context).

Variants: the production call (FusedSDPA with the full bias), the same on a finer context bucket, and sliced
FusedSDPA (context slices without bias except where bucket padding can reach, online softmax merge) in BF16 and
dynamic FP8. Error is measured against an FP32 reference on a subset of query rows.
Run with the service stopped (the device is exclusive):  python prefill_attn_bench.py [quick]"""
import math
import sys
import time
import types

import torch
import habana_frameworks.torch as htorch  # noqa: F401
import habana_frameworks.torch.core as htcore  # noqa: F401
from habana_frameworks.torch.hpex.kernels import FusedSDPA, fp8_fused_sdpa

import vllm_gaudi.extension.utils as vutils

_cfg = types.SimpleNamespace(enable_fsdpa_slicing=False, VLLM_HPU_FSDPA_DYNAMIC_FP8=False, merged_prefill=False,
                             use_bucketing=True, fp32_softmax=False)
vutils.get_config = lambda: _cfg

DEV = "hpu"
Q, HEADS, KV, D, BLOCK = 2048, 24, 4, 256, 128
SCALE = D**-0.5
FP8 = torch.float8_e4m3fn
REF_ROWS = 192  # last rows of the chunk (they see the most keys) + first rows


def make_case(ctx_real, ctx_pad_blocks, peaked, seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx_pad = ctx_pad_blocks * BLOCK
    kv_len = ctx_pad + Q
    kscale = 1.6 if peaked else 0.5
    k = (torch.randn(1, KV, kv_len, D, generator=g) * kscale).to(FP8).to(torch.bfloat16)
    v = (torch.randn(1, KV, kv_len, D, generator=g) * 0.5).to(FP8).to(torch.bfloat16)
    q = (torch.randn(1, HEADS, Q, D, generator=g) * 1.5).to(torch.bfloat16)
    # padding blocks hold whatever the pad page holds: make it loud so a missing mask shows up
    k[:, :, ctx_real:ctx_pad] = 4.0
    v[:, :, ctx_real:ctx_pad] = 8.0
    bias = torch.zeros(1, 1, Q, kv_len, dtype=torch.bfloat16)
    bias[..., ctx_real:ctx_pad] = -math.inf
    bias[..., ctx_pad:] = torch.triu(torch.full((Q, Q), -math.inf), diagonal=1).to(torch.bfloat16)
    return [t.to(DEV) for t in (q, k, v, bias)], ctx_pad


def reference(q, k, v, bias):
    rows = torch.cat([torch.arange(Q - REF_ROWS + 32, Q), torch.arange(0, 32)]).to(DEV)
    out = torch.empty(1, HEADS, rows.numel(), D, dtype=torch.float32, device=DEV)
    b = bias[0, 0].index_select(0, rows).float()
    group = HEADS // KV
    for h in range(KV):
        kh, vh = k[0, h].float(), v[0, h].float()
        qh = q[0, h * group:(h + 1) * group].index_select(1, rows).float()
        s = torch.matmul(qh, kh.t()) * SCALE + b
        out[0, h * group:(h + 1) * group] = torch.matmul(torch.softmax(s, dim=-1), vh)
    return rows, out


def bf16_unsliced(q, k, v, bias):
    """ModuleFusedSDPA without slicing (BF16 KV cache)."""
    return FusedSDPA.apply(q, k, v, bias, 0.0, False, SCALE, "fast", True, None, "right", False, False, (-1, -1),
                           None)


def fp8_unsliced(prob_scale):
    """ModuleFP8FusedSDPA without slicing: the production call with an FP8 KV cache and no QUANT_CONFIG (all
    scales 1, so softmax probabilities are cast to FP8 unscaled). prob_scale != 1 scales them first."""
    mod = vutils.ModuleFP8FusedSDPA(fp8_fused_sdpa)
    mod.scale_amax = torch.tensor(float(prob_scale), dtype=torch.float32)
    mod.descale_amax = torch.tensor(1.0 / prob_scale, dtype=torch.float32)

    def fn(q, k, v, bias):
        return mod(q, k, v, bias, 0.0, True, SCALE, "fast", True, None, "right")

    return fn


def _slice_setup(mod, chunk, masked_ctx_chunks, dynamic_fp8=False):
    mod.enable_slicing, mod.adaptive, mod.dynamic_fp8 = True, False, dynamic_fp8
    mod.slice_thld, mod.chunk_size, mod._with_graph_breaks = chunk, chunk, False
    mod.num_padded_query_chunks, mod.num_padded_ctx_chunks = math.ceil(Q / chunk), masked_ctx_chunks


def sliced(chunk, masked_ctx_chunks, kind, prob_scale=1.0):
    """kind: "bf16" (SlicedFusedSDPA), "dynfp8" (SlicedFusedSDPA, per-call FP8 scales, probs x128) or "fp8"
    (ModuleFP8FusedSDPA's SlicedFP8FusedSDPA with static unit scales, probs x prob_scale)."""
    if kind == "fp8":
        mod = vutils.ModuleFP8FusedSDPA(fp8_fused_sdpa)
        mod.scale_amax = torch.tensor(float(prob_scale), dtype=torch.float32)
        mod.descale_amax = torch.tensor(1.0 / prob_scale, dtype=torch.float32)
        _slice_setup(mod._sliced_module, chunk, masked_ctx_chunks)

        def fn(q, k, v, bias):
            return mod(q, k, v, bias, 0.0, True, SCALE, "fast", True, None, "right")

        return fn
    mod = vutils.SlicedFusedSDPA.__new__(vutils.SlicedFusedSDPA)
    torch.nn.Module.__init__(mod)
    _slice_setup(mod, chunk, masked_ctx_chunks, kind == "dynfp8")

    def fn(q, k, v, bias):
        return mod(q, k, v, bias, 0.0, True, SCALE, "fast")

    return fn


def two_call(variant):
    """Fixed number of kernels at any context length: queries against the cached context without a bias (padding
    excluded by valid_seq_len, or by a bias when variant == "ctxbias"), queries against their own chunk with
    is_causal, then the online-softmax merge of the two partial results."""
    from habana_frameworks.torch.hpex.kernels.FusedSDPA import gqa_input_reshape_fwd, gqa_output_reshape

    def part(q, k, v, mask, causal, valid):
        res = torch.ops.hpu.sdpa_recomp_fwd(q, k, v, mask, 0.0, SCALE, causal, True, "fast", valid, "right")
        return tuple(gqa_output_reshape(x).to(torch.float32) for x in res[:3])

    def fn(q, k, v, bias, ctx_real):
        q_len = q.shape[-2]
        prefix = k.shape[-2] - q_len
        qg, kg, vg, bg = gqa_input_reshape_fwd(q, k, v, bias)
        kc, vc = kg[..., :prefix, :], vg[..., :prefix, :]
        if variant == "ctxbias":
            o1, m1, l1 = part(qg, kc, vc, bg[..., :prefix], False, None)
        else:
            o1, m1, l1 = part(qg, kc, vc, None, False, ctx_real)
        o2, m2, l2 = part(qg, kg[..., prefix:, :], vg[..., prefix:, :], None, True, None)
        l1, l2 = l1 * 128.0, l2 * 128.0  # fast softmax returns 1/sum scaled by 1/128 (as in SlicedFusedSDPA)
        m = torch.maximum(m1, m2)
        w1 = torch.exp(m1 - m) / l1
        w2 = torch.exp(m2 - m) / l2
        return ((w1 * o1 + w2 * o2) / (w1 + w2)).to(q.dtype)

    return fn


def timed(fn, inp, iters=10):
    out = fn(*inp)
    torch.hpu.synchronize()
    fn(*inp)
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*inp)
    torch.hpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3, out


def run(name, cfn, inp, ref):
    rows, want = ref
    t0 = time.perf_counter()
    try:
        ms, out = timed(cfn, inp)
    except Exception as e:  # report and keep going
        print(f"    {name:36s} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return
    got = out.index_select(2, rows).float()
    err = ((got - want).norm() / want.norm()).item()
    fin = bool(torch.isfinite(out).all().item())
    print(f"    {name:36s} {ms:8.2f} ms  rel err {err:.4f}  finite {fin}  ({time.perf_counter() - t0:.0f} s)",
          flush=True)


def variants(max_pad_blocks, quick):
    v = [("production: fp8 FSDPA + bias", fp8_unsliced(1.0))]
    if max_pad_blocks is None:
        return v
    v += [("fp8 FSDPA + bias, probs x128", fp8_unsliced(128.0)), ("bf16 FSDPA + bias", bf16_unsliced)]
    for chunk in ((16384, ) if quick else (16384, 32768)):
        masked = math.ceil(max_pad_blocks * BLOCK / chunk)
        v += [(f"sliced fp8 chunk {chunk}", sliced(chunk, masked, "fp8")),
              (f"sliced fp8 chunk {chunk}, probs x128", sliced(chunk, masked, "fp8", 128.0)),
              (f"sliced bf16 chunk {chunk}", sliced(chunk, masked, "bf16")),
              (f"sliced dyn-fp8 chunk {chunk}", sliced(chunk, masked, "dynfp8"))]
    return [(name, torch.compile(fn, backend="hpu_backend", dynamic=False)) for name, fn in v]


def twocall_bench():
    for ctx_real, blocks in [(95_000, 768), (180_000, 1408), (95_000, 1017)]:
        inp, ctx_pad = make_case(ctx_real, blocks, True)
        ref = reference(*inp)
        valid = torch.tensor([ctx_real], dtype=torch.int32, device=DEV)
        print(f"== context {ctx_real}, {blocks} blocks, peaked scores", flush=True)
        run("bf16 FSDPA + bias (production)", torch.compile(bf16_unsliced, backend="hpu_backend", dynamic=False), inp, ref)
        run(f"sliced bf16 chunk 8192", torch.compile(sliced(8192, 1, "bf16"), backend="hpu_backend", dynamic=False), inp, ref)
        for variant in ("valid_len", "ctxbias"):
            fn = two_call(variant)
            cfn = torch.compile(lambda q, k, v, b, fn=fn: fn(q, k, v, b, valid), backend="hpu_backend", dynamic=False)
            run(f"two-call {variant}", cfn, inp, ref)
        del inp, ref
        torch.hpu.synchronize()


def split_bench():
    """Batched-slice split attention, compiled once with dynamic shapes and run at three context lengths."""
    from vllm_gaudi.extension import prefill_split_attn as split
    import torch._dynamo.utils as dutils
    cfn = torch.compile(lambda q, k, v, b: split.split_attention(q, k, v, b, SCALE), backend="hpu_backend",
                        dynamic=True)
    for ctx_real, blocks in [(95_000, 768), (180_000, 1408), (250_000, 1984), (60_000, 512)]:
        inp, ctx_pad = make_case(ctx_real, blocks, True)
        ref = reference(*inp)
        print(f"== context {ctx_real}, {blocks} blocks, peaked scores", flush=True)
        run("bf16 FSDPA + bias (production)", torch.compile(bf16_unsliced, backend="hpu_backend", dynamic=False), inp,
            ref)
        t0 = time.perf_counter()
        run("split (batched slices, dynamic)", cfn, inp, ref)
        print(f"    graphs so far: {dict(dutils.counters['stats'])}  first call incl. compile "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
        inp2, _ = make_case(ctx_real, blocks, False, seed=1)
        run("split, flat scores", cfn, inp2, reference(*inp2))
        del inp, ref, inp2
        torch.hpu.synchronize()


def split_sweep():
    """Time the dynamic split graph across slice counts; static compile at the odd ones for comparison."""
    from vllm_gaudi.extension import prefill_split_attn as split
    fn = lambda q, k, v, b: split.split_attention(q, k, v, b, SCALE)
    cfn = torch.compile(fn, backend="hpu_backend", dynamic=True)
    for blocks in (768, 1024, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1728, 1856, 1984):
        inp, ctx_pad = make_case(blocks * BLOCK - 3000, blocks, True)
        ref = reference(*inp)
        n = ctx_pad // split.SLICE - 1
        run(f"dynamic split, {blocks} blocks, n={n}", cfn, inp, ref)
        if blocks in (1344, 1408, 1472):
            run(f"static split, {blocks} blocks", torch.compile(fn, backend="hpu_backend", dynamic=False), inp, ref)
        del inp, ref
        torch.hpu.synchronize()


def split_parts():
    """Which piece of the split graph is slow under dynamic shapes."""
    from vllm_gaudi.extension import prefill_split_attn as split
    C = split.SLICE

    def prep(q, k, v, b):
        prefix = k.shape[-2] - q.shape[-2]
        n = (prefix - C) // C
        qg = q.reshape(1, KV, HEADS // KV, q.shape[-2], D)
        kg = k.reshape(1, KV, 1, k.shape[-2], D)[..., :prefix - C, :].reshape(1, KV, n, C, D)
        return qg, kg.permute(0, 2, 1, 3, 4).reshape(n, KV, 1, C, D), n

    def p_qrep(q, k, v, b):
        qg, _, n = prep(q, k, v, b)
        return qg.expand(n, KV, HEADS // KV, q.shape[-2], D).contiguous()

    def p_kstack(q, k, v, b):
        return prep(q, k, v, b)[1].contiguous()

    def p_sdpa(q, k, v, b):
        qg, ks, n = prep(q, k, v, b)
        qr = qg.expand(n, KV, HEADS // KV, q.shape[-2], D).contiguous()
        return torch.ops.hpu.sdpa_recomp_fwd(qr, ks, ks, None, 0.0, SCALE, False, True, "fast", None, "right")[0]

    parts = [("q expand", p_qrep), ("k stack", p_kstack), ("batched sdpa", p_sdpa),
             ("full split", lambda q, k, v, b: split.split_attention(q, k, v, b, SCALE))]
    compiled = [(name, torch.compile(fn, backend="hpu_backend", dynamic=True)) for name, fn in parts]
    for blocks in (768, 1408, 1472, 1536, 1600, 1728):
        inp, _ = make_case(blocks * BLOCK - 3000, blocks, True)
        for name, cfn in compiled:
            ms, _ = timed(cfn, inp, iters=5)
            print(f"    {blocks} blocks  {name:14s} {ms:9.2f} ms", flush=True)
        del inp
        torch.hpu.synchronize()


def sweep():
    """BF16 only: the production call on both buckets and sliced BF16 at several slice sizes."""
    for ctx_real, exp_blocks, fine_blocks in [(95_000, 1017, 768), (180_000, 2047, 1408)]:
        for label, blocks in (("exp bucket", exp_blocks), ("fine bucket", fine_blocks)):
            v = [("bf16 FSDPA + bias (production)", bf16_unsliced)]
            if label == "fine bucket":
                v += [(f"sliced bf16 chunk {c}", sliced(c, math.ceil(64 * BLOCK / c), "bf16")) for c in (4096, 8192, 16384)]
            inp, ctx_pad = make_case(ctx_real, blocks, True)
            ref = reference(*inp)
            print(f"== context {ctx_real}, {label} {blocks} blocks, peaked scores", flush=True)
            for name, fn in v:
                run(name, torch.compile(fn, backend="hpu_backend", dynamic=False), inp, ref)
            del inp, ref
            torch.hpu.synchronize()


def main():
    if "sweep" in sys.argv:
        return sweep()
    if "splitparts" in sys.argv:
        return split_parts()
    if "splitsweep" in sys.argv:
        return split_sweep()
    if "split" in sys.argv:
        return split_bench()
    if "twocall" in sys.argv:
        return twocall_bench()
    quick = "quick" in sys.argv
    # (real context tokens, production exponential bucket, fine bucket (64-block steps from 256))
    cases = [(95_000, 1017, 768)] if quick else [(95_000, 1017, 768), (180_000, 2047, 1408), (250_000, 2047, 1984)]
    for ctx_real, exp_blocks, fine_blocks in cases:
        for label, blocks, max_pad_blocks in (("exp bucket", exp_blocks, None), ("fine bucket", fine_blocks, 64)):
            compiled = variants(max_pad_blocks, quick)
            for peaked in (False, True):
                inp, ctx_pad = make_case(ctx_real, blocks, peaked)
                ref = reference(*inp)
                print(f"== context {ctx_real}, {label} {blocks} blocks ({ctx_pad} keys + {Q}), "
                      f"{'peaked' if peaked else 'flat'} scores", flush=True)
                for name, cfn in compiled:
                    run(name, cfn, inp, ref)
                del inp, ref
                torch.hpu.synchronize()


if __name__ == "__main__":
    main()
