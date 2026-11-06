# SPDX-License-Identifier: Apache-2.0
"""H22 Phase 1 — paired A/B of the production prefill entries + two probes.

``bench_h22_prefill_moe.py`` measures each candidate against a fresh baseline,
which is fine for 20-40 % effects but not for the 2-6 % this unit is chasing:
run-to-run spread on one routed-MoE call is +-3 %, so a single median from a
single order cannot separate the two paths.

This script measures the SHIPPED entry points (``prefill_grouped_h18`` against
``prefill_grouped_h22``, both including their own plan) in **interleaved
pairs**, alternating which one runs first, and reports the median of the
per-pair ratios. Drift, clock and allocator state cancel inside each pair.

Two probes ride along:

``dequant_scaling``
    dequant of ``n`` experts for n = 64..U in ONE launch per projection. This
    is the "is the unpack proportional to the expert count" question the whole
    tail-offload idea rests on.

``tail_probe``
    the 2-B split taken apart: the dense part of the layer alone (a plan
    truncated at ``n_dense``) vs the same plus the tail on the fused GEMV, at
    the exact ``S`` the tail holds. It shows where the tail's cost really is.

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h22_ab.py \
      --json tools/qwen4exp_flashnext/bench_h22_ab_eager.json
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
os.environ.setdefault("VLLM_QWEN4EXP_MOE_H15", "1")

sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(REPO / "tools/qwen4exp_flashnext"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import habana_frameworks.torch.core  # noqa: E402,F401

from bench_h22_prefill_moe import (  # noqa: E402
    TOP_K, _drain, _h9a, _plan_for, _score, load_routed_planes, real_routing)


def _sync() -> None:
    torch.hpu.synchronize()


def _peak(fn) -> float:
    _drain(fn())
    _sync()
    torch.hpu.reset_peak_memory_stats()
    _drain(fn())
    return torch.hpu.max_memory_allocated() / 2 ** 20


def _timed(fn) -> float:
    _sync()
    start = time.perf_counter()
    out = fn()
    _drain(out)
    return (time.perf_counter() - start) * 1e3


def ab(fn_a, fn_b, pairs: int = 11, warm: int = 2) -> dict:
    """Interleaved paired A/B: median ms each, and the median paired ratio."""
    for _ in range(warm):
        _drain(fn_a())
        _drain(fn_b())
    _sync()
    left, right, ratios = [], [], []
    for index in range(pairs):
        order = ((fn_a, left), (fn_b, right)) if index % 2 == 0 else ((fn_b, right), (fn_a, left))
        pair = {}
        for function, bucket in order:
            elapsed = _timed(function)
            bucket.append(elapsed)
            pair[id(function)] = elapsed
        ratios.append(pair[id(fn_a)] / pair[id(fn_b)])
    return {
        "pairs": pairs,
        "a_ms": statistics.median(left), "b_ms": statistics.median(right),
        "a_min_ms": min(left), "b_min_ms": min(right),
        "ratio_median": statistics.median(ratios),
        "ratio_p25": statistics.quantiles(ratios, n=4)[0] if pairs >= 4 else None,
        "ratio_p75": statistics.quantiles(ratios, n=4)[2] if pairs >= 4 else None,
        "b_wins": sum(1 for r in ratios if r > 1.0),
    }


def probe_dequant_scaling(store, selected, count, device, log) -> None:
    """Dequant cost vs number of experts in ONE launch per projection."""
    from vllm_gaudi.models.qwen4exp.packed_moe import materialized_ids

    rows = []
    for n in (64, 128, 192, 256, 320, count):
        if n > count:
            continue
        ids = materialized_ids(selected[:n], device)

        def call():
            return (store.dequant("gate_proj", ids), store.dequant("up_proj", ids),
                    store.dequant("down_proj", ids))

        ms = statistics.median([_timed(call) for _ in range(3)])
        rows.append({"experts": n, "ms": ms, "us_per_expert": 1e3 * ms / n})
        print(f"  dequant 3x1launch n={n:3d}: {ms:6.2f} ms ({1e3 * ms / n:5.2f} us/expert)",
              flush=True)
    log.setdefault("dequant_scaling", []).append({"unique": count, "rows": rows})


def probe_tail(hidden, topk_ids, topk_weights, store, plan, chunk, tail_k, device, log) -> None:
    """Dense part alone vs dense + fused-GEMV tail, at the tail's exact S."""
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped_h22, gemv_slots, materialized_ids, swiglu_gemv_slots)

    count = int(plan["count"])
    slots = hidden.shape[0] * TOP_K
    counts_sorted = plan["counts_sorted"]
    n_dense = int((counts_sorted > tail_k).sum().item())
    dense_slots = int(counts_sorted[:n_dense].sum().item())
    if dense_slots >= slots:
        print(f"  tail<={tail_k}: no expert has count <= {tail_k} at this regime "
              f"(min count {int(counts_sorted[count - 1])}) -> nothing to offload", flush=True)
        log.setdefault("tail_probe", []).append(
            {"tail_k": tail_k, "tail_experts": 0, "tail_slots": 0, "skipped": True})
        return
    dense_plan = dict(plan)
    dense_plan["count"] = n_dense
    dense_plan["starts"] = list(plan["starts"][:(n_dense + chunk - 1) // chunk]) + [dense_slots]
    tail_slots = slots - dense_slots

    def dense_only():
        return apply_routed_experts_grouped_h22(
            hidden, plan["local"], topk_weights, store, plan["selected_by_rank"],
            dense_plan, mix=False)

    slot_out = dense_only()
    token_all = torch.arange(hidden.shape[0], device=device).repeat_interleave(TOP_K)
    weights_flat = topk_weights.reshape(-1)
    tail_index = plan["slot_order"][dense_slots:slots]
    tail_ids = plan["selected_by_rank"].index_select(
        0, plan["local_sorted"][dense_slots:slots]).to(torch.int32)
    x_tail = hidden.index_select(0, token_all.index_select(0, tail_index))

    def gemv_only():
        act = swiglu_gemv_slots(store, tail_ids, x_tail)
        if act is None:
            act = F.silu(gemv_slots(store, "gate_proj", tail_ids, x_tail)) \
                * gemv_slots(store, "up_proj", tail_ids, x_tail)
        down = gemv_slots(store, "down_proj", tail_ids, act)
        slot_out[dense_slots:slots] = down * weights_flat.index_select(
            0, tail_index).unsqueeze(-1)
        return slot_out

    def dense_plus_gemv():
        out = dense_only()
        act = swiglu_gemv_slots(store, tail_ids, x_tail)
        if act is None:
            act = F.silu(gemv_slots(store, "gate_proj", tail_ids, x_tail)) \
                * gemv_slots(store, "up_proj", tail_ids, x_tail)
        down = gemv_slots(store, "down_proj", tail_ids, act)
        out[dense_slots:slots] = down * weights_flat.index_select(
            0, tail_index).unsqueeze(-1)
        return out

    def full_h22():
        from vllm_gaudi.models.qwen4exp.packed_moe import apply_routed_experts_grouped_h22 as run
        return run(hidden, plan["local"], topk_weights, store, plan["selected_by_rank"], plan)

    dense_ms = statistics.median([_timed(dense_only) for _ in range(3)])
    gemv_ms = statistics.median([_timed(gemv_only) for _ in range(3)])
    both_ms = statistics.median([_timed(dense_plus_gemv) for _ in range(3)])
    full_ms = statistics.median([_timed(full_h22) for _ in range(3)])
    row = {
        "tail_k": tail_k, "tail_experts": count - n_dense, "tail_slots": tail_slots,
        "dense_experts": n_dense, "dense_slots": dense_slots,
        "dense_only_ms": dense_ms, "gemv_only_ms": gemv_ms,
        "dense_plus_gemv_ms": both_ms, "full_h22_ms": full_ms,
        "offload_delta_ms": both_ms - full_ms,
    }
    log.setdefault("tail_probe", []).append(row)
    print(f"  tail<={tail_k}: {row['tail_experts']} experts / {tail_slots} slots -> "
          f"dense-only {dense_ms:6.2f} + gemv {gemv_ms:5.2f} = {both_ms:6.2f} ms vs "
          f"no-split {full_ms:6.2f} ms (delta {row['offload_delta_ms']:+.2f})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=REPO / "tools/qwen4exp_flashnext/bench_h22_ab_eager.json")
    parser.add_argument("--layers", type=str, default="0,3")
    parser.add_argument("--tokens", type=str, default="512,2048")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--supers", type=str, default="1,4")
    parser.add_argument("--orders", type=str, default="chunk")
    parser.add_argument("--tails", type=str, default="1,4")
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if EXT is None or not KERNEL.is_file():
        raise SystemExit("build the TPC kernels + pytorch extension first")
    torch.ops.load_library(str(EXT))
    if not torch.hpu.is_available():
        raise SystemExit("no HPU")

    from vllm_gaudi.models.qwen4exp.packed_moe import (
        prefill_grouped_h18, prefill_grouped_h22)

    log: dict = {"unit": "H22 Phase 1 paired A/B (production prefill entries)",
                 "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
                 "pairs": args.pairs, "layers": {}}
    for layer in (int(v) for v in args.layers.split(",") if v.strip()):
        store = load_routed_planes(layer)
        _sync()
        log["layers"][f"L{layer}"] = {}
        for tokens in (int(v) for v in args.tokens.split(",") if v.strip()):
            hidden, topk_ids, topk_weights = real_routing(layer, tokens, args.seed)
            plan = _plan_for(topk_ids, args.chunk, store)
            entry: dict = {"unique": int(plan["count"]),
                           "cap": int(plan["caps"][0]), "slots": tokens * TOP_K}
            print(f"\n== L{layer} T={tokens}: U={entry['unique']} cap={entry['cap']} ==", flush=True)

            reference = _h9a(hidden, topk_ids, topk_weights, store)
            probe_dequant_scaling(store, plan["selected_by_rank"], int(plan["count"]),
                                  hidden.device, entry)
            for tail_k in (int(v) for v in args.tails.split(",") if v.strip()):
                probe_tail(hidden, topk_ids, topk_weights, store, plan, args.chunk,
                           tail_k, hidden.device, entry)

            def h18():
                # The H18 entry body is shared with the H22 bucket hoist, so the
                # baseline must switch this unit OFF to be H18 as recorded.
                os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22"] = "0"
                return prefill_grouped_h18(hidden, topk_ids, topk_weights, store)

            os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22"] = "0"
            assert h18() is not None
            os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22"] = "1"
            for order in (v.strip() for v in args.orders.split(",") if v.strip()):
                os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22_ORDER"] = order
                for n_super in (int(v) for v in args.supers.split(",") if v.strip()):
                  os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22_SUPERS"] = str(n_super)

                  def h22():
                      os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22"] = "1"
                      return prefill_grouped_h22(hidden, topk_ids, topk_weights, store)

                  h22_out = h22()
                  if h22_out is None:
                      print(f"  supers={n_super}: H22 declined", flush=True)
                      continue
                  score_h18 = _score(h18(), reference)
                  score_pair = _score(h22_out, h18())
                  row = ab(h18, h22, pairs=args.pairs)
                  row.update({
                      "supers": n_super, "order": os.environ.get(
                          "VLLM_QWEN4EXP_MOE_PREFILL_H22_ORDER", "chunk"),
                      "h18_peak_MiB": _peak(h18), "h22_peak_MiB": _peak(h22),
                      "h22_vs_h9a": score_h18["bf16_bit_diff"],
                      "h22_vs_h18": score_pair["bf16_bit_diff"],
                      "h22_vs_h18_elements": score_pair["bf16_elements"],
                      "cosine_vs_h18": score_pair["cosine"],
                  })
                  entry.setdefault("ab", []).append(row)
                  print(f"  [{order} supers={n_super}] H18 {row['a_ms']:6.2f} ms | "
                        f"H22 {row['b_ms']:6.2f} ms | ratio {row['ratio_median']:.4f} (p25 {row['ratio_p25']:.4f} "
                        f"p75 {row['ratio_p75']:.4f}, H22 faster in {row['b_wins']}/{row['pairs']}) "
                        f"| bitdiff vs H18 {score_pair['bf16_bit_diff']}/"
                        f"{score_pair['bf16_elements']}", flush=True)
            os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22"] = "1"
            os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H22_SUPERS"] = "1"
            log["layers"][f"L{layer}"][f"T={tokens}"] = entry
            args.json.write_text(json.dumps(log, indent=2) + "\n")
            del hidden, topk_ids, topk_weights, reference, plan
        del store
    args.json.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
