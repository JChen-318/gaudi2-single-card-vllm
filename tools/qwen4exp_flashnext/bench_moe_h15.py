# SPDX-License-Identifier: Apache-2.0
"""H15 Phase 1: standalone MoE-block micro-benchmark + accuracy checker.

Synthetic/random packed NVFP4 weights, real layer-0 geometry, no checkpoint:

  hidden=2560  E=512  top_k=10  moe_inter=640  shared_inter=640

Components (exactly the three costs left after H14):

  Router        F.linear [512,2560] -> topk -> softmax(topk).float -> bf16/int32
  SharedExpert  gate_up [1280,2560] -> SiluAndMul -> down [2560,640],
                sigmoid(shared_gate [1,2560] @ x) * down_out
  RoutedMoE     NVFP4 indexed GEMV gate + up -> F.silu(gate)*up ->
                NVFP4 indexed GEMV down -> * topk_weights -> sum over top_k

H15 paths timed side by side (when the built extension provides them):

  Router        F.linear + custom_qwen4exp_moe_router_topk_bf16_gaudi2
  SharedExpert  merged [1281,2560] linear (gate_up ++ shared_gate, built once
                in memory) -> custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2
                (silu*up, then * sigmoid(shared_logit), scalar-linearity move)
                -> down linear                                   (3 launches)
  RoutedMoE     custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2
                (x tile loaded once, gate+up accumulated together,
                BF16-rounded gate/up, silu+mul in kernel) -> down GEMV ->
                custom_qwen4exp_moe_down_epilogue_bf16_gaudi2
                (BF16 weighted product per slot, F32 slot-order sum)

Bit-exactness (BF16 bit patterns, not tolerances) is checked for the fused
SwiGLU GEMV, the router top-k and the down epilogue. Two HPU facts shape
the references: Habana's BF16 silu/sigmoid kernels are approximate (their
output differs from exact silu in ~40% of elements by 1 ulp, probed in
isolation), so the SwiGLU reference is the exact chain (standalone GEMV
BF16 outputs, CPU silu+mul) -- the fused kernel matches it 0-differing and
is *more* accurate than the HPU chain it replaces (<= 1 ulp from it); and
HPU torch.topk's tie order is partition-dependent, so the router check is
value-sequence + weights bit-exactness with every differing id verified
to be an exact logit tie. The shared expert is checked with the MoE-block
tolerance (rel < 2e-2, cosine > 0.999): moving the per-token sigmoid
scalar in front of the (linear) down projection is exact in real
arithmetic but relocates one BF16 rounding.

At T=512 the routed experts stay on the production prefill path
(dequant-unique + expert-major grouped bmm): S = T*top_k = 5120 slots is far
beyond the decode GEMV's job, and a GEMV there would re-read every expert
per slot instead of reusing it across tokens. H15 therefore reports the
prefill routed number unchanged and only accelerates router + shared.

Run (eager, synchronized attribution -- the H14 component methodology):

  GC_KERNEL_PATH=$PWD/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so \
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_moe_h15.py \
      --json tools/qwen4exp_flashnext/bench_moe_h15.json

  ... --check-only     accuracy only, no timing
  ... --tokens 1,4     decode only
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"), None)
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import numpy as np  # noqa: E402

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
MOE_INTER = 640
SHARED_INTER = 640


def _median(xs):
    return statistics.median(xs)


def time_batched(fn, iters, samples):
    """Median ms/call, many calls between synchronizes (unsync pipeline).

    Synchronized per-call timing (time_sync) attributes a host round-trip
    to every launch, which is how the H14 component table was built; this
    batched form is the closer model of an unsynchronized decode step and
    of lazy graph replay, where launches queue back to back.
    """
    import torch

    def first(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    acc = first(fn())
    torch.hpu.synchronize()
    _ = float(acc.float().sum().item())
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _i in range(iters):
            out = first(fn())
            acc = out if acc is None else acc + out
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / iters)
    _ = float(acc.float().sum().item())
    return _median(times)


def time_sync(fn, warmup, samples):
    """Median ms/call with a synchronize around every call (attribution)."""
    import torch

    check = 0.0
    for _ in range(warmup):
        out = fn()
        torch.hpu.synchronize()
        check += float(out.float().sum().item()) if isinstance(out, torch.Tensor) else 0.0
    times = []
    for _ in range(samples):
        torch.hpu.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        if isinstance(out, torch.Tensor):
            check += float(out.float().sum().item())
    if check != check or abs(check) == float("inf"):
        raise SystemExit(f"non-finite checksum {check}")
    return _median(times)


def _diff_bits(a, b):
    import torch

    a_c, b_c = a.cpu(), b.cpu()
    if a_c.dtype == torch.bfloat16:
        return int((a_c.view(torch.uint16) != b_c.view(torch.uint16)).sum().item()), a_c.numel()
    return int((a_c != b_c).sum().item()), a_c.numel()


def _max_ulp(a, b):
    """Largest BF16 spacing-step between two same-shaped BF16 tensors."""
    import torch

    def ordered(t):
        u = t.cpu().view(torch.uint16).to(torch.int32)
        return torch.where(u >= 32768, 65535 - u, u + 32768)

    return int((ordered(a) - ordered(b)).abs().max().item())


def make_planes(rng, n, k):
    packed = rng.integers(0, 256, size=(EXPERTS, n, k // 2), dtype=np.uint8)
    scales = rng.integers(0, 256, size=(EXPERTS, n, k // 16), dtype=np.uint8)
    bad = (scales & 0x7F) == 0x7F
    scales = np.where(bad, scales & 0x7E, scales).astype(np.uint8)
    scale2 = (rng.random(EXPERTS, dtype=np.float32) * 1e-4 + 1e-5).astype(np.float32)
    return packed, scales, scale2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--tokens", type=str, default="1,4,512")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    tokens_list = [int(t) for t in args.tokens.split(",") if t.strip()]

    if not KERNEL.is_file() or EXT is None:
        raise SystemExit("build kernels + pytorch extension first (see bench_decode_moe.py)")

    import torch
    import torch.nn.functional as F

    import habana_frameworks.torch.core  # noqa: F401

    from vllm_gaudi.models.qwen4exp.packed_moe import materialized_ids

    torch.ops.load_library(str(EXT))
    ns = torch.ops.custom_op
    gemv = ns.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2
    dequant = ns.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    swiglu_op = getattr(ns, "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2", None)
    router_op = getattr(ns, "custom_qwen4exp_moe_router_topk_bf16_gaudi2", None)
    shared_op = getattr(ns, "custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2", None)
    down_epi_op = getattr(ns, "custom_qwen4exp_moe_down_epilogue_bf16_gaudi2", None)
    h15_ops = dict(swiglu=swiglu_op is not None, router=router_op is not None,
                   shared=shared_op is not None, down_epilogue=down_epi_op is not None)
    print(f"bench_moe_h15: lazy={os.environ.get('PT_HPU_LAZY_MODE')} h15_ops={h15_ops}")

    rng = np.random.default_rng(20261103)
    print("building synthetic planes E=512 (gate/up/down) ...", flush=True)

    def to_hpu(packed, scales, scale2):
        return (torch.from_numpy(packed).to("hpu"), torch.from_numpy(scales).to("hpu"),
                torch.from_numpy(scale2).to("hpu"))

    planes = {}
    for kind, (n, k) in (("gate", (MOE_INTER, HIDDEN)), ("up", (MOE_INTER, HIDDEN)),
                         ("down", (HIDDEN, MOE_INTER))):
        planes[kind] = to_hpu(*make_planes(rng, n, k))
    gen = torch.Generator().manual_seed(7)

    def rand_bf16(*shape, scale=0.04):
        return ((torch.rand(*shape, generator=gen) - 0.5) * 2 * scale).bfloat16()

    router_w = rand_bf16(EXPERTS, HIDDEN, scale=0.02).to("hpu")
    gate_up_w = rand_bf16(2 * SHARED_INTER, HIDDEN, scale=0.02).to("hpu")
    down_w = rand_bf16(HIDDEN, SHARED_INTER, scale=0.02).to("hpu")
    shared_gate_w = rand_bf16(1, HIDDEN, scale=0.02).to("hpu")
    merged_w = torch.cat([gate_up_w, shared_gate_w], dim=0).contiguous() if shared_op else None

    try:
        from vllm.model_executor.layers.activation import SiluAndMul
        silu_and_mul = SiluAndMul()
    except Exception:
        silu_and_mul = None

    results = {"benchmark": "bench_moe_h15", "geometry": dict(
        hidden=HIDDEN, experts=EXPERTS, top_k=TOP_K, moe_inter=MOE_INTER,
        shared_inter=SHARED_INTER), "h15_ops": h15_ops, "tokens": {}}
    failures = []

    for tokens in tokens_list:
        slots = tokens * TOP_K
        print(f"\n===== T={tokens} S={slots} =====", flush=True)
        hidden = rand_bf16(tokens, HIDDEN).to("hpu")
        topk_ids = torch.randint(0, EXPERTS, (tokens, TOP_K), generator=gen,
                                 dtype=torch.int32).to("hpu")
        topk_w = torch.rand(tokens, TOP_K, generator=gen).bfloat16().to("hpu")
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).contiguous()
        flat_ids = materialized_ids(topk_ids, torch.device("hpu"))
        decode = slots <= 512

        # ---------------- baseline closures --------------------------------
        def route_base(h=hidden):
            logits = F.linear(h, router_w)
            values, idx = torch.topk(logits, TOP_K, dim=-1)
            w = torch.softmax(values.float(), dim=-1).to(h.dtype)
            return idx.to(torch.int32), w

        def shared_base(h=hidden):
            gate_up, _ = (F.linear(h, gate_up_w), None)
            if silu_and_mul is not None:
                act = silu_and_mul(gate_up)
            else:
                act = F.silu(gate_up[:, :SHARED_INTER]) * gate_up[:, SHARED_INTER:]
            out = F.linear(act, down_w)
            return torch.sigmoid(F.linear(h, shared_gate_w)) * out

        def grouped_base(h=hidden, ids=topk_ids, w=topk_w):
            fid = materialized_ids(ids, torch.device("hpu"))
            gate = gemv(fid, *planes["gate"][:2], h, planes["gate"][2])
            up = gemv(fid, *planes["up"][:2], h, planes["up"][2])
            act = F.silu(gate) * up
            d = gemv(fid, *planes["down"][:2], act.contiguous(), planes["down"][2])
            weighted = d * w.reshape(-1, 1).to(d.dtype)
            return (weighted.sum(dim=0, keepdim=True) if tokens == 1
                    else weighted.view(tokens, TOP_K, HIDDEN).sum(dim=1))

        def routed_prefill(h=hidden, ids=topk_ids, w=topk_w):
            from vllm_gaudi.models.qwen4exp.packed_moe import (
                apply_routed_experts_grouped, compact_expert_ids)

            class _E:  # minimal dequant shim over the three planes
                def dequant(self, kind, u):
                    p, s, s2 = planes[{"gate_proj": "gate", "up_proj": "up",
                                       "down_proj": "down"}[kind]]
                    return dequant(materialized_ids(u, torch.device("hpu")), p, s, s2)

            unique, local = compact_expert_ids(ids, EXPERTS)
            e = _E()
            return apply_routed_experts_grouped(
                h, local, w, e.dequant("gate_proj", unique),
                e.dequant("up_proj", unique), e.dequant("down_proj", unique))

        # ---------------- H15 closures --------------------------------------
        def route_h15(h=hidden):
            return router_op(F.linear(h, router_w), TOP_K)

        def shared_h15(h=hidden):
            merged = F.linear(h, merged_w)
            act = shared_op(merged)
            return F.linear(act, down_w)

        def grouped_h15(h=hidden, ids=topk_ids, w=topk_w):
            fid = materialized_ids(ids, torch.device("hpu"))
            act = swiglu_op(fid, planes["gate"][0], planes["gate"][1], planes["gate"][2],
                            planes["up"][0], planes["up"][1], planes["up"][2], h)
            d = gemv(fid, *planes["down"][:2], act, planes["down"][2])
            return down_epi_op(d, w)

        # ---------------- accuracy ------------------------------------------
        acc = {}
        if tokens == 1 or tokens == 4 or args.check_only:
            pass
        # Router (all T)
        base_ids, base_w = route_base()
        torch.hpu.synchronize()
        if router_op is not None:
            got_ids, got_w = route_h15()
            torch.hpu.synchronize()
            d_ids, _ = _diff_bits(got_ids, base_ids)
            d_w, n_w = _diff_bits(got_w, base_w)
            # Value-sequence bit-exactness: the logits at the fused ids, in
            # order, must be exactly torch.topk's values. Every differing id
            # is then an exact BF16 logit tie (HPU torch.topk's tie order is
            # partition-dependent, not an index order -- see the kernel).
            logits_acc = F.linear(hidden, router_w)
            vals_ref, _ = torch.topk(logits_acc, TOP_K, dim=-1)
            torch.hpu.synchronize()
            got_vals = torch.gather(logits_acc.cpu(), 1, got_ids.cpu().long())
            d_vals, _ = _diff_bits(got_vals, vals_ref.cpu())
            acc["router"] = dict(ids_differing=d_ids, weights_differing=d_w,
                                  values_differing=d_vals, weights_numel=n_w)
            print(f"  acc router: ids differing {d_ids} (exact-tie only), "
                  f"values differing {d_vals}, weights differing {d_w}/{n_w}")
            if d_vals or d_w:
                failures.append(
                    f"T={tokens} router values/weights not bit-exact vals={d_vals} w={d_w}/{n_w}")
        # Shared (all T)
        ref_shared = shared_base()
        torch.hpu.synchronize()
        if shared_op is not None:
            got_shared = shared_h15()
            torch.hpu.synchronize()
            gap = (got_shared.float().cpu() - ref_shared.float().cpu()).abs().max().item()
            scale = ref_shared.float().cpu().abs().max().clamp_min(1e-6).item()
            cos = torch.nn.functional.cosine_similarity(
                got_shared.float().cpu().reshape(1, -1),
                ref_shared.float().cpu().reshape(1, -1)).item()
            acc["shared"] = dict(rel=gap / scale, cosine=cos)
            print(f"  acc shared: rel {gap/scale:.3e} cosine {cos:.6f}")
            if not (gap / scale < 2e-2 and cos > 0.999):
                failures.append(f"T={tokens} shared diverged rel={gap/scale:.3e}")
        # Routed decode (S <= 512)
        if decode and swiglu_op is not None:
            fid = flat_ids
            gate = gemv(fid, *planes["gate"][:2], hidden, planes["gate"][2])
            torch.hpu.synchronize()
            up = gemv(fid, *planes["up"][:2], hidden, planes["up"][2])
            torch.hpu.synchronize()
            ref_act_hpu = F.silu(gate) * up
            torch.hpu.synchronize()
            # Exact reference: Habana's BF16 silu/sigmoid kernels are
            # approximate (their own output differs from exact silu in ~40%
            # of BF16 elements at these magnitudes, by 1 ulp -- probed in
            # isolation), so bit-exactness is judged against the exact
            # chain: BF16 gate/up from the standalone GEMVs, silu+mul on
            # CPU (exact at BF16). The fused kernel is the accurate one.
            ref_act = (F.silu(gate.cpu()) * up.cpu()).to("hpu")
            got_act = swiglu_op(fid, planes["gate"][0], planes["gate"][1], planes["gate"][2],
                                planes["up"][0], planes["up"][1], planes["up"][2], hidden)
            torch.hpu.synchronize()
            d_act, n_act = _diff_bits(got_act, ref_act)
            d_hpu, _ = _diff_bits(got_act, ref_act_hpu)
            ulp_hpu = _max_ulp(got_act, ref_act_hpu)
            acc["swiglu"] = dict(differing_vs_exact=d_act,
                                  differing_vs_hpu_chain=d_hpu, max_ulp_vs_hpu_chain=ulp_hpu,
                                  numel=n_act)
            print(f"  acc swiglu_gemv vs exact chain: differing {d_act}/{n_act} "
                  f"({'100% bit-identical' if d_act == 0 else 'MISMATCH'})")
            print(f"  acc swiglu_gemv vs HPU silu chain: differing {d_hpu}/{n_act} "
                  f"(max {ulp_hpu} ulp; HPU BF16 silu is itself approximate)")
            if d_act or ulp_hpu > 4:
                failures.append(
                    f"T={tokens} swiglu vs exact differing {d_act}/{n_act}, ulp {ulp_hpu}")
            if down_epi_op is not None:
                dwn = gemv(fid, *planes["down"][:2], ref_act.contiguous(), planes["down"][2])
                torch.hpu.synchronize()
                ref_routed = (dwn * topk_w.reshape(-1, 1).to(dwn.dtype))
                ref_routed = (ref_routed.sum(dim=0, keepdim=True) if tokens == 1
                              else ref_routed.view(tokens, TOP_K, HIDDEN).sum(dim=1))
                torch.hpu.synchronize()
                got_routed = grouped_h15()
                torch.hpu.synchronize()
                d_r, n_r = _diff_bits(got_routed, ref_routed)
                acc["routed_full"] = dict(differing=d_r, numel=n_r)
                print(f"  acc routed_full (swiglu+down+epilogue): differing {d_r}/{n_r} "
                      f"({'100% bit-identical' if d_r == 0 else 'MISMATCH'})")
                if d_r:
                    failures.append(f"T={tokens} routed_full differing {d_r}/{n_r}")
        results["tokens"][str(tokens)] = {"accuracy": acc}
        if args.check_only:
            continue

        # ---------------- sub-op timing -------------------------------------
        samples = args.samples if tokens < 512 else min(3, args.samples)
        warmup = args.warmup if tokens < 512 else 1
        rows = {}

        def T(name, fn):
            ms = time_sync(fn, warmup, samples)
            rows[name] = ms
            print(f"  {name:34s} {ms:9.4f} ms")
            return ms

        # Router sub-ops
        logits = F.linear(hidden, router_w)
        values, idx = torch.topk(logits, TOP_K, dim=-1)
        torch.hpu.synchronize()
        T("router.linear", lambda: F.linear(hidden, router_w))
        T("router.topk_on_logits", lambda: torch.topk(logits, TOP_K, dim=-1)[0])
        T("router.softmax_cast_chain", lambda: (
            torch.softmax(values.float(), dim=-1).to(hidden.dtype), idx.to(torch.int32)))
        T("router.base_full", route_base)
        if router_op is not None:
            T("router.fused_topk_on_logits", lambda: router_op(logits, TOP_K))
            T("router.h15_full", route_h15)
        # Shared sub-ops
        gate_up_out = F.linear(hidden, gate_up_w)
        torch.hpu.synchronize()
        T("shared.gate_up_linear", lambda: F.linear(hidden, gate_up_w))
        if silu_and_mul is not None:
            T("shared.silu_and_mul", lambda: silu_and_mul(gate_up_out))
            act_pre = silu_and_mul(gate_up_out)
        else:
            T("shared.silu_mul", lambda: F.silu(gate_up_out[:, :SHARED_INTER])
              * gate_up_out[:, SHARED_INTER:])
            act_pre = F.silu(gate_up_out[:, :SHARED_INTER]) * gate_up_out[:, SHARED_INTER:]
        torch.hpu.synchronize()
        T("shared.down_linear", lambda: F.linear(act_pre, down_w))
        sg = F.linear(hidden, shared_gate_w)
        down_pre = F.linear(act_pre, down_w)
        torch.hpu.synchronize()
        T("shared.gate_linear", lambda: F.linear(hidden, shared_gate_w))
        T("shared.sigmoid_mul", lambda: torch.sigmoid(sg) * down_pre)
        T("shared.base_full", shared_base)
        if shared_op is not None:
            T("shared.merged_linear_1281", lambda: F.linear(hidden, merged_w))
            merged_out = F.linear(hidden, merged_w)
            torch.hpu.synchronize()
            T("shared.h15_epilogue", lambda: shared_op(merged_out))
            act_h15 = shared_op(merged_out)
            torch.hpu.synchronize()
            T("shared.h15_down_linear", lambda: F.linear(act_h15, down_w))
            T("shared.h15_full", shared_h15)
        # Routed sub-ops
        if decode:
            act_ref = F.silu(gemv(flat_ids, *planes["gate"][:2], hidden, planes["gate"][2])) \
                * gemv(flat_ids, *planes["up"][:2], hidden, planes["up"][2])
            torch.hpu.synchronize()
            slot_down = gemv(flat_ids, *planes["down"][:2], act_ref.contiguous(),
                             planes["down"][2])
            torch.hpu.synchronize()
            T("routed.gate_gemv", lambda: gemv(
                flat_ids, *planes["gate"][:2], hidden, planes["gate"][2]))
            T("routed.up_gemv", lambda: gemv(
                flat_ids, *planes["up"][:2], hidden, planes["up"][2]))
            gate_pre = gemv(flat_ids, *planes["gate"][:2], hidden, planes["gate"][2])
            up_pre = gemv(flat_ids, *planes["up"][:2], hidden, planes["up"][2])
            torch.hpu.synchronize()
            T("routed.silu_mul", lambda: F.silu(gate_pre) * up_pre)
            T("routed.down_gemv", lambda: gemv(
                flat_ids, *planes["down"][:2], act_ref.contiguous(), planes["down"][2]))
            T("routed.weighted_mul", lambda: slot_down * topk_w.reshape(-1, 1).to(slot_down.dtype))
            wm = slot_down * topk_w.reshape(-1, 1).to(slot_down.dtype)
            torch.hpu.synchronize()
            T("routed.view_sum", lambda: (wm.sum(dim=0, keepdim=True) if tokens == 1
                                          else wm.view(tokens, TOP_K, HIDDEN).sum(dim=1)))
            T("routed.base_grouped_gemv", grouped_base)
            if swiglu_op is not None:
                T("routed.h15_swiglu_gemv", lambda: swiglu_op(
                    flat_ids, planes["gate"][0], planes["gate"][1], planes["gate"][2],
                    planes["up"][0], planes["up"][1], planes["up"][2], hidden))
                if down_epi_op is not None:
                    T("routed.h15_down_epilogue", lambda: down_epi_op(slot_down, topk_w))
                    T("routed.h15_grouped", grouped_h15)
        else:
            T("routed.prefill_grouped", routed_prefill)
        # Full MoE block
        def moe_base(h=hidden):
            ids_, w_ = route_base(h)
            routed = grouped_base(h, ids_, w_) if decode else routed_prefill(h, ids_, w_)
            return routed + shared_base(h)

        T("moe_block.base_full", moe_base)
        if decode and swiglu_op is not None and router_op is not None and shared_op is not None:
            def moe_h15(h=hidden):
                ids_, w_ = route_h15(h)
                return grouped_h15(h, ids_, w_) + shared_h15(h)

            T("moe_block.h15_full", moe_h15)
        elif not decode and router_op is not None and shared_op is not None:
            def moe_h15_512(h=hidden):
                ids_, w_ = route_h15(h)
                return routed_prefill(h, ids_, w_) + shared_h15(h)

            T("moe_block.h15_full_prefill_routed", moe_h15_512)
        if decode:
            # Unsynchronized (batched) forms of the headline comparisons:
            # sync timing charges every custom-op launch a host round-trip
            # that a queued decode step / lazy graph does not pay per op.
            def B(name, fn):
                ms = time_batched(fn, iters=20, samples=3)
                rows[name + "_batched"] = ms
                print(f"  {name + '_batched':34s} {ms:9.4f} ms")
                return ms

            B("router.base_full", route_base)
            if router_op is not None:
                B("router.h15_full", route_h15)
            B("shared.base_full", shared_base)
            if shared_op is not None:
                B("shared.h15_full", shared_h15)
            B("routed.base_grouped_gemv", grouped_base)
            if swiglu_op is not None and down_epi_op is not None:
                B("routed.h15_grouped", grouped_h15)
            B("moe_block.base_full", moe_base)
            if swiglu_op is not None and router_op is not None and shared_op is not None:
                B("moe_block.h15_full", moe_h15)
        results["tokens"][str(tokens)]["timing_ms"] = rows
        # Per-48-layer projection for the three components
        for comp, base_key, h15_key in (
                ("Router", "router.base_full", "router.h15_full"),
                ("SharedExpert", "shared.base_full", "shared.h15_full"),
                ("RoutedMoE", "routed.base_grouped_gemv", "routed.h15_grouped")):
            if base_key in rows:
                line = f"  x48 {comp:14s} base {48*rows[base_key]:8.2f} ms"
                if h15_key in rows:
                    line += f"  h15 {48*rows[h15_key]:8.2f} ms  speedup {rows[base_key]/rows[h15_key]:.2f}x"
                print(line)
            b_base, b_h15 = base_key + "_batched", h15_key + "_batched"
            if b_base in rows and b_h15 in rows:
                print(f"  x48 {comp + ' (batched)':22s} base {48*rows[b_base]:8.2f} ms  "
                      f"h15 {48*rows[b_h15]:8.2f} ms  speedup {rows[b_base]/rows[b_h15]:.2f}x")

    results["failures"] = failures
    print("\nFAILURES:", failures if failures else "none")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
