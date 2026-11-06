# SPDX-License-Identifier: Apache-2.0
"""H27 Part 2 — prefill MoE chunk size: interleaved paired A/B on a real layer.

H18's default is a fixed ``chunk = 64`` experts per dequant+bucket+bmm step. Its
win is the per-chunk tight ``cap`` on the bucket rows; its cost is
``3 * n_chunks`` dequant launches, one bucket per chunk and ~17 eager dispatches
per chunk. H27 makes each dequant launch 2.2-2.4x cheaper, so the balance at the
small token counts (``T <= 512``, where ``cap`` is small and the padding H18
removes is tiny) has to be re-measured.

Measured per (layer, T, chunk):
  * ``sync_ms``    median of ``--samples`` launches, each surrounded by a sync
  * ``batched_ms`` median of a 3-deep pipelined burst (one sync)
  * paired A/B     interleaved pairs against the production default (chunk 64)
                   and against the H9(a) path (``..._PREFILL_H18=0``), median of
                   the per-pair ratios (H22 trap 1: a single-order reading of a
                   ~1 % effect is noise)
  * bucket rows / pad fraction / BF16 peak / bit difference against the
    chunk=512 (or H9(a)) reference

Both layers matter: layer 0 routing is skewed (a few experts take most slots),
layer 3 is close to uniform, so the ``cap`` profile -- and therefore the chunk
crossover -- differs.

Run (card must be free, one process per layer):

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h27_chunk.py \
      --layer 0 --json /tmp/h27/chunk_L0_eager.json
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h27_chunk.py \
      --layer 3 --json /tmp/h27/chunk_L3_eager.json
"""
from __future__ import annotations

import argparse
import gc
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

import habana_frameworks.torch.core  # noqa: E402,F401

import bench_h18_prefill_moe as harness  # noqa: E402
from bench_h18_prefill_moe import (  # noqa: E402
    BF16_MIB_PER_EXPERT, EXPERTS, HIDDEN, MOE_INTER, TOP_K, _drain, _score,
    peak_mib, time_batched, time_sync,
)

CHUNK_ENV = "VLLM_QWEN4EXP_MOE_PREFILL_H18_CHUNK"
H18_ENV = "VLLM_QWEN4EXP_MOE_PREFILL_H18"


def _timed(fn) -> float:
    torch.hpu.synchronize()
    start = time.perf_counter()
    out = fn()
    _drain(out)
    return (time.perf_counter() - start) * 1e3


def ab(fn_a, fn_b, pairs: int, warm: int = 2) -> dict:
    """Interleaved paired A/B; ratio = a_ms / b_ms (> 1 means b is faster)."""
    for _ in range(warm):
        _drain(fn_a())
        _drain(fn_b())
    torch.hpu.synchronize()
    left, right, ratios = [], [], []
    for index in range(pairs):
        order = ((fn_a, left), (fn_b, right)) if index % 2 == 0 else (
            (fn_b, right), (fn_a, left))
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
        "b_wins": sum(1 for ratio in ratios if ratio > 1.0),
    }


def _chunk_plan(hidden, topk_ids, topk_weights, store, chunk):
    """(bucket rows, per-chunk caps, chunks) for one chunk size -- shape only."""
    from vllm_gaudi.models.qwen4exp.packed_moe import count_ordered_plan

    _selected, _local, plan = count_ordered_plan(topk_ids, store.num_experts, chunk)
    if plan is None or plan["count"] == 0:
        return None
    n_chunks = (plan["count"] + chunk - 1) // chunk
    rows = sum(min(chunk, plan["count"] - i * chunk) * int(plan["caps"][i])
               for i in range(n_chunks))
    return {"chunks": n_chunks, "bucket_rows": rows,
            "caps": [int(c) for c in plan["caps"]], "count": int(plan["count"])}


def chunked_entry(hidden, topk_ids, topk_weights, store, chunk):
    """The H18 body without ``PREFILL_H18_MIN_SLOTS``.

    ``prefill_grouped_h18`` refuses everything below 1024 slots (T=64 is 640),
    so the T=64 row of the sweep has to call the same two functions the entry
    calls. Above the gate this is exactly what the production entry runs.
    """
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped_chunked, count_ordered_plan)

    selected, local, plan = count_ordered_plan(topk_ids, store.num_experts, chunk)
    if plan is None or plan["count"] == 0:
        return None
    return apply_routed_experts_grouped_chunked(
        hidden, local, topk_weights, store, selected, plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=str, default="64,128,256,512,1024,2048")
    parser.add_argument("--chunks", type=str, default="64,128,256,512")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--pairs", type=int, default=9)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if EXT is None or not KERNEL.is_file():
        raise SystemExit("build the TPC kernels + pytorch extension first")
    torch.ops.load_library(str(EXT))
    if not torch.hpu.is_available():
        raise SystemExit("no HPU")

    harness.LAYER = f"model.language_model.layers.{args.layer}.mlp"
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        PREFILL_H18_MIN_SLOTS, prefill_grouped_h18)

    print(f"\nloading layer-{args.layer} routed planes ({harness.LAYER}) ...", flush=True)
    started = time.time()
    store = harness.load_routed_planes()
    torch.hpu.synchronize()
    print(f"  ready in {time.time() - started:.0f}s", flush=True)

    chunks = [int(v) for v in args.chunks.split(",") if v.strip()]
    tokens_list = [int(v) for v in args.tokens.split(",") if v.strip()]
    log = {
        "unit": "H27 Part 2 prefill chunk sweep (H27 dequant kernel active)",
        "layer": args.layer, "layer_prefix": harness.LAYER,
        "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "min_slots_h18": PREFILL_H18_MIN_SLOTS,
        "geometry": {"hidden": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K,
                     "moe_inter": MOE_INTER,
                     "bf16_MiB_per_expert_all_3_proj": BF16_MIB_PER_EXPERT},
        "chunks": chunks, "tokens": tokens_list, "pairs": args.pairs,
        "regimes": {},
    }

    for tokens in tokens_list:
        key = f"T={tokens}"
        hidden, topk_ids, topk_weights = harness.real_routing(tokens, args.seed)
        slots = tokens * TOP_K
        entry: dict = {"slots": slots, "variants": {}, "ab_vs_chunk64": {},
                       "ab_vs_h9a": {}, "plans": {}}
        print(f"\n== L{args.layer} {key}: slots={slots} ==", flush=True)

        # H9(a) baseline: the path the block runs when the H18 entry declines
        # (``prefill_grouped_h18`` returns None with ``..._PREFILL_H18=0``, so
        # the H9(a) primitives are called directly, as bench_h18_prefill_moe
        # does). It is the timing A/B baseline, not an accuracy reference.
        def run_h9a():
            return harness._baseline(hidden, topk_ids, topk_weights, store)

        h9a_ms = time_sync(run_h9a, samples=args.samples)
        h9a_peak, _ = peak_mib(run_h9a)
        entry["variants"]["h9a"] = {"sync_ms": h9a_ms, "peak_MiB": h9a_peak}

        # Accuracy reference: the H18 body with chunk = E (dedup only, no
        # chunking), which is the same code path with one group.
        reference = chunked_entry(hidden, topk_ids, topk_weights, store, EXPERTS)
        print(f"  H9(a) {h9a_ms:7.2f} ms | reference = H18 chunk=E "
              f"{'declined' if reference is None else 'ok'}", flush=True)

        functions = {}
        for chunk in chunks:

            def run(chunk=chunk):
                return chunked_entry(hidden, topk_ids, topk_weights, store, chunk)

            out = run()
            declined = out is None
            plan = _chunk_plan(hidden, topk_ids, topk_weights, store, chunk)
            entry["plans"][f"chunk_{chunk}"] = plan
            score = _score(out, reference) if reference is not None else None
            sync_ms = 0.0 if declined else time_sync(run, samples=args.samples)
            batched_ms = 0.0 if declined else time_batched(run, inner=3)
            peak, _ = peak_mib(run)
            functions[chunk] = run
            entry["variants"][f"chunk_{chunk}"] = {
                "declined": declined, "sync_ms": sync_ms, "batched_ms": batched_ms,
                "peak_MiB": peak, "accuracy_vs_chunkE": score,
                "chunks": plan["chunks"] if plan else 0,
                "bucket_rows": plan["bucket_rows"] if plan else 0,
                "pad_frac": (1.0 - slots / plan["bucket_rows"]) if plan and plan["bucket_rows"] else None,
                "cap_max": max(plan["caps"]) if plan else 0,
                "materialized_MiB": 2 * min(chunk, plan["count"] if plan else 0)
                * BF16_MIB_PER_EXPERT,
            }
            print(f"  chunk={chunk:4d}: sync {sync_ms:7.2f} ms batched {batched_ms:7.2f} "
                  f"peak {peak:6.0f} MiB chunks={entry['variants'][f'chunk_{chunk}']['chunks']:2d} "
                  f"rows={entry['variants'][f'chunk_{chunk}']['bucket_rows']:6d} "
                  f"cap={entry['variants'][f'chunk_{chunk}']['cap_max']:4d} "
                  f"bitdiff={(score or {}).get('bf16_bit_diff')}"
                  f"{' DECLINED' if declined else ''}", flush=True)

        for chunk in chunks:
            if chunk == chunks[0] or entry["variants"][f"chunk_{chunk}"]["declined"]:
                continue
            row = ab(functions[chunks[0]], functions[chunk], pairs=args.pairs)
            row["chunk"] = chunk
            entry["ab_vs_chunk64"][f"chunk_{chunk}"] = row
            print(f"  A/B chunk{chunks[0]} vs chunk{chunk}: "
                  f"{row['a_ms']:7.2f} | {row['b_ms']:7.2f} ms  ratio "
                  f"{row['ratio_median']:.4f} (p25 {row['ratio_p25']:.4f} "
                  f"p75 {row['ratio_p75']:.4f}, larger chunk faster "
                  f"{row['b_wins']}/{row['pairs']})", flush=True)

        for chunk in chunks:
            if entry["variants"][f"chunk_{chunk}"]["declined"]:
                continue
            row = ab(run_h9a, functions[chunk], pairs=args.pairs)
            row["chunk"] = chunk
            entry["ab_vs_h9a"][f"chunk_{chunk}"] = row
            print(f"  A/B H9a vs chunk{chunk}: {row['a_ms']:7.2f} | "
                  f"{row['b_ms']:7.2f} ms  ratio {row['ratio_median']:.4f} "
                  f"(H18 faster {row['b_wins']}/{row['pairs']})", flush=True)

        log["regimes"][key] = entry
        args.json.write_text(json.dumps(log, indent=2) + "\n")
        del hidden, topk_ids, topk_weights, reference
        gc.collect()

    args.json.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
