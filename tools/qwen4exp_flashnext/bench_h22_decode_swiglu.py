# SPDX-License-Identifier: Apache-2.0
"""H22 decode: fused SwiGLU GEMV vs 2 x GEMV + ``silu * up``.

H21 Phase 2 left one open item: the fused ``swiglu_gemv_slots`` kernel does two
passes per member (gate, then up), so it has half the member-level parallelism
of two standalone GEMV launches. Measured then (real layer-0 planes):

    S=10  fused 0.1605 ms  vs  2 x gemv 0.1610 ms   -> fused, +0.3 %
    S=40  fused 0.4849 ms  vs  2 x gemv 0.4419 ms   -> 2 x gemv, -8.9 %
    S=80  fused 0.9421 ms  vs  2 x gemv 0.8409 ms   -> 2 x gemv, -10.7 %

The 6-9 % at S >= 40 is real but **not lossless**: the fused kernel evaluates
silu in F32 with an accurate exp + Newton reciprocal (bit-identical to exact
silu, H15), while the two-launch chain must use the device BF16 ``F.silu``,
whose hardware sigmoid moves ~29 % of activations by one BF16 ulp. An explicit
F32 emulation still differs on 26 % because HPU's F32 sigmoid is the same
approximation. The shipped default therefore stays
``SWIGLU_GEMV_MAX_SLOTS = 512`` (always fused); ``10`` is an operator opt-in
that trades the 1-ulp decode change for the 2 x GEMV form above S=10.

This script measures both forms and the production ``grouped_gemv`` at
T=1/4/8 with the gate at 512 (always fused) vs 10 (the opt-in):

  * ``slot_curve`` gives the per-S timing and the BF16 bit differences;
  * ``grouped_gemv`` is the production decode entry (T=1 stays fused at
    S=10 with either gate; T=4/8 take the 2 x GEMV form when gated);
  * ``grouped_bit_diff`` records the output difference the opt-in would cause.

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h22_decode_swiglu.py \
      --json tools/qwen4exp_flashnext/bench_h22_decode_swiglu_eager.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"), None)
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()

os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
os.environ.setdefault("VLLM_QWEN4EXP_MOE_H15", "1")

sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(REPO / "tools/qwen4exp_flashnext"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import habana_frameworks.torch.core  # noqa: E402,F401

from bench_h22_prefill_moe import (  # noqa: E402
    HIDDEN, TOP_K, load_routed_planes, real_routing, time_batched, time_sync)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=REPO / "tools/qwen4exp_flashnext/bench_h22_decode_swiglu_eager.json")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=str, default="1,4,8")
    parser.add_argument("--slots", type=str, default="10,40,80")
    args = parser.parse_args()
    if EXT is None or not KERNEL.is_file():
        raise SystemExit("build the TPC kernels + pytorch extension first")
    torch.ops.load_library(str(EXT))
    if not torch.hpu.is_available():
        raise SystemExit("no HPU")

    from vllm_gaudi.models.qwen4exp.packed_moe import (
        gemv_slots, grouped_gemv, materialized_ids, swiglu_gemv_slots)

    print(f"loading layer-{args.layer} routed planes ...", flush=True)
    store = load_routed_planes(args.layer)
    torch.hpu.synchronize()
    log: dict = {"unit": "H22 Phase 1 decode: swiglu_gemv_slots gate",
                 "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"), "layer": args.layer,
                 "slot_curve": [], "grouped_gemv": []}

    # ---- kernel-level A/B: the two SwiGLU forms, per S --------------------
    for s in (int(v) for v in args.slots.split(",") if v.strip()):
        gen = torch.Generator().manual_seed(20261105 + s)
        ids = torch.randint(0, store.num_experts, (s,), generator=gen).to(torch.int32).to("hpu")
        x = (torch.randn(s, HIDDEN, generator=gen) * 0.5).bfloat16().to("hpu")
        ids_m = materialized_ids(ids, "hpu")

        def fused():
            return swiglu_gemv_slots(store, ids_m, x)

        def two():
            # plain BF16 chain: F.silu runs on the device's BF16 silu kernel
            return F.silu(gemv_slots(store, "gate_proj", ids_m, x)) \
                * gemv_slots(store, "up_proj", ids_m, x)

        def two_f32():
            # the fused kernel's contract: silu in F32 on the BF16-rounded gate,
            # rounded to BF16, then multiplied by the BF16 up value and rounded.
            gate = gemv_slots(store, "gate_proj", ids_m, x)
            up = gemv_slots(store, "up_proj", ids_m, x)
            return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

        left = fused()
        right = two()
        emul = two_f32()
        diff = int((left.detach().cpu().view(torch.uint16)
                    != right.detach().cpu().view(torch.uint16)).sum().item())
        diff_emul = int((left.detach().cpu().view(torch.uint16)
                         != emul.detach().cpu().view(torch.uint16)).sum().item())
        row = {
            "S": s, "bit_diff": diff, "bit_diff_f32_emul": diff_emul,
            "elements": int(left.numel()),
            "fused_ms": time_sync(fused, samples=5),
            "two_gemv_ms": time_sync(two, samples=5),
            "two_gemv_f32_ms": time_sync(two_f32, samples=5),
            "fused_batched_ms": time_batched(fused, inner=4, samples=3),
            "two_gemv_batched_ms": time_batched(two, inner=4, samples=3),
            "two_gemv_f32_batched_ms": time_batched(two_f32, inner=4, samples=3),
        }
        row["delta_pct"] = 100.0 * (row["two_gemv_ms"] - row["fused_ms"]) / row["fused_ms"]
        row["delta_f32_pct"] = 100.0 * (row["two_gemv_f32_ms"] - row["fused_ms"]) / row["fused_ms"]
        log["slot_curve"].append(row)
        print(f"  S={s:3d}: fused {row['fused_ms']:.4f} ms | 2xgemv {row['two_gemv_ms']:.4f} "
              f"({row['delta_pct']:+.1f}%) | 2xgemv+f32silu {row['two_gemv_f32_ms']:.4f} "
              f"({row['delta_f32_pct']:+.1f}%) | batched {row['fused_batched_ms']:.4f} / "
              f"{row['two_gemv_batched_ms']:.4f} / {row['two_gemv_f32_batched_ms']:.4f} | "
              f"bitdiff bf16silu {diff}/{row['elements']}, f32silu {diff_emul}/"
              f"{row['elements']}", flush=True)

    # ---- production entry: grouped_gemv at T=1/4/8 ----------------------
    for tokens in (int(v) for v in args.tokens.split(",") if v.strip()):
        hidden, topk_ids, topk_weights = real_routing(args.layer, tokens)
        results = {}
        for label, gate in (("fused_always", "512"), ("gated_s10", "10")):
            os.environ["VLLM_QWEN4EXP_MOE_SWIGLU_GEMV_MAX_SLOTS"] = gate
            out = grouped_gemv(hidden, topk_ids, topk_weights, store)
            results[label] = out
            row = {
                "T": tokens, "S": tokens * TOP_K, "max_slots": int(gate),
                "sync_ms": time_sync(lambda: grouped_gemv(hidden, topk_ids, topk_weights, store),
                                     samples=5),
                "batched_ms": time_batched(
                    lambda: grouped_gemv(hidden, topk_ids, topk_weights, store), inner=4, samples=3),
            }
            log["grouped_gemv"].append(row)
            print(f"  T={tokens} S={tokens * TOP_K:3d} gate={gate:>3s}: "
                  f"sync {row['sync_ms']:.4f} ms batched {row['batched_ms']:.4f} ms", flush=True)
        os.environ["VLLM_QWEN4EXP_MOE_SWIGLU_GEMV_MAX_SLOTS"] = "10"
        diff = int((results["fused_always"].detach().cpu().view(torch.uint16)
                    != results["gated_s10"].detach().cpu().view(torch.uint16)).sum().item())
        log.setdefault("grouped_bit_diff", []).append(
            {"T": tokens, "bit_diff": diff, "elements": int(results["fused_always"].numel())})
        print(f"    grouped_gemv bitdiff (fused-always vs gated) {diff}/"
              f"{results['fused_always'].numel()}", flush=True)

    args.json.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
