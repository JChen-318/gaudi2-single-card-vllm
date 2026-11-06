# SPDX-License-Identifier: Apache-2.0
"""H18 Step 2 / Part 2 — Direction 2-A: prefill MoE HBM round-trip & bucketing.

Real Layer-0 weights, real Layer-0 routing, T in {64, 128, 512, 2048},
E=512, top_k=10, H=2560, I=640. Nothing is served, nothing is profiled.

What is measured
----------------
``parts`` (H9(a) baseline, ``VLLM_QWEN4EXP_MOE_PREFILL_H18=0``):

  router -> ``compact_expert_ids`` (presence mask + argsort + ONE host read)
        -> ``dequant`` gate/up/down for every unique expert (4.2 GiB BF16 at
           U=427: 3 tensors of ``[U, I, H]`` / ``[U, H, I]``)
        -> ``apply_routed_experts_grouped``: a SECOND presence matrix for
           ``cap = int(counts.max())`` (second host read), a THIRD one inside
           ``_add_grouped_bucket``, one bucket of ``[U * cap, H]`` rows where
           ``cap`` is the single busiest expert's count, three ``bmm`` calls,
           one ``index_add_``.

Candidates (each timed against the same baseline in the same process):

  2-A(i)   hybrid sparse-expert GEMV + dense dequant/bmm:
           experts with ``1 <= count <= K_sparse`` never enter the BF16
           materialization; their slots go through the fused NVFP4 GEMV
           (``swiglu_gemv_slots`` + ``gemv_slots`` down). Swept
           K_sparse = 0, 1, 2, 3, 4, 6, 8.
  2-A(ii)  expert-chunked dequant + per-chunk cap (auto tiering, because the
           experts are ordered by descending count before chunking). Swept
           chunk = 8, 16, 32, 64, 128, 256, 512. Also reports the bucket rows
           and the BF16 peak.
  2-A(iii) one presence matrix + one host transfer for the whole shape plan
           (present count, per-chunk caps, per-chunk slot offsets).

Accuracy: every candidate is compared with the baseline output on the same
weights/inputs (cosine, rel max, BF16 bit differences). The routed dot
products are unchanged, so only the order of the per-slot accumulation into
the output differs.

Run (card must be free; eager first, then lazy in a separate process):

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h18_prefill_moe.py \
      --json tools/qwen4exp_flashnext/bench_h18_prefill_moe_eager.json
  PT_HPU_LAZY_MODE=1 .venv/bin/python tools/qwen4exp_flashnext/bench_h18_prefill_moe.py \
      --json tools/qwen4exp_flashnext/bench_h18_prefill_moe_lazy.json

  ... --only parts   # dissection + candidate sweeps only
  ... --only block   # Qwen4ExpSparseMoeBlock Before/After only
  ... --tokens 512   # one regime
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"), None)
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.0.mlp"

os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
os.environ.setdefault("VLLM_QWEN4EXP_MOE_H15", "1")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import habana_frameworks.torch.core  # noqa: E402,F401

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
MOE_INTER = 640
MAX_GEMV_SLOTS = 512  # swiglu GEMV kernel S cap
# gate [I, H] + up [I, H] + down [H, I], all BF16: 9.375 MiB = 9.83 MB
BF16_MIB_PER_EXPERT = (3 * MOE_INTER * HIDDEN * 2) / 2 ** 20

_SAFETENSORS_DTYPE = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
                      "I8": torch.int8, "F16": torch.float16, "I32": torch.int32}
_HANDLES: dict = {}
_HEADERS: dict = {}
_WEIGHT_MAP: dict | None = None


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def _weight_map() -> dict:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        _WEIGHT_MAP = json.loads(
            (CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    return _WEIGHT_MAP


def _load(name: str) -> torch.Tensor:
    shard = _weight_map()[name]
    if shard not in _HEADERS:
        path = CHECKPOINT / "tensors" / shard
        with open(path, "rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_bytes))
        _HEADERS[shard] = (header, 8 + header_bytes, path)
    header, base, path = _HEADERS[shard]
    info = header[name]
    handle = _HANDLES.get(shard)
    if handle is None:
        handle = open(path, "rb")
        _HANDLES[shard] = handle
    start, end = info["data_offsets"]
    handle.seek(base + start)
    buf = handle.read(end - start)
    if info["dtype"] == "F8_E4M3":
        tensor = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        tensor = torch.frombuffer(bytearray(buf), dtype=_SAFETENSORS_DTYPE[info["dtype"]])
    return tensor.reshape(info["shape"])


def load_routed_planes(device: str = "hpu"):
    """The three routed-expert planes of layer 0, packed, on the device."""
    from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts

    store = PackedRoutedExperts(EXPERTS)
    for kind, (n, k) in (("gate_proj", (MOE_INTER, HIDDEN)),
                         ("up_proj", (MOE_INTER, HIDDEN)),
                         ("down_proj", (HIDDEN, MOE_INTER))):
        packed = torch.empty(EXPERTS, n, k // 2, dtype=torch.uint8)
        scales = torch.empty(EXPERTS, n, k // 16, dtype=torch.uint8)
        scale2 = torch.empty(EXPERTS, dtype=torch.float32)
        for expert in range(EXPERTS):
            base = f"{LAYER}.experts.{expert}.{kind}"
            packed[expert] = _load(base + ".weight").view(torch.uint8).reshape(n, k // 2)
            scales[expert] = _load(base + ".weight_scale").view(torch.uint8).reshape(n, k // 16)
            scale2[expert] = float(_load(base + ".weight_scale_2").reshape(-1)[0])
        setattr(store, f"{kind}_packed", packed.to(device))
        setattr(store, f"{kind}_scale", scales.to(device))
        setattr(store, f"{kind}_scale2", scale2.to(device))
    return store


def real_routing(tokens: int, seed: int = 20261105):
    """Layer-0 router on a seeded hidden state (the production router GEMM)."""
    gate_weight = _load(f"{LAYER}.gate.weight").to("hpu").bfloat16()
    gen = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).bfloat16().to("hpu")
    logits = F.linear(hidden, gate_weight)
    values, indices = torch.topk(logits, TOP_K, dim=-1)
    weights = torch.softmax(values.float(), dim=-1).bfloat16()
    return hidden, indices.to(torch.int32), weights


# ---------------------------------------------------------------------------
# timing / memory
# ---------------------------------------------------------------------------
def _is_lazy() -> bool:
    return os.environ.get("PT_HPU_LAZY_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


def _mark_step() -> None:
    if not _is_lazy():
        return
    try:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()
    except Exception:  # noqa: BLE001
        pass


def _drain(out) -> float:
    if out is None or isinstance(out, (int, float)):
        return float(out or 0.0)
    if isinstance(out, (tuple, list)):
        return sum(_drain(item) for item in out)
    _mark_step()
    torch.hpu.synchronize()
    return float(out.reshape(-1)[0].float().item()) if out.numel() else 0.0


def time_sync(fn, warm: int = 1, samples: int = 3) -> float:
    """Median ms with a synchronize around every call (launch attribution)."""
    for _ in range(warm):
        _drain(fn())
    times = []
    for _ in range(samples):
        _mark_step()
        torch.hpu.synchronize()
        start = time.perf_counter()
        out = fn()
        check = _drain(out)
        times.append((time.perf_counter() - start) * 1e3)
    if check != check or abs(check) == float("inf"):
        raise SystemExit(f"non-finite checksum {check}")
    return statistics.median(times)


def time_batched(fn, inner: int = 4, samples: int = 3) -> float:
    """Median ms per call with ``inner`` calls queued and one sync (throughput)."""
    for _ in range(2):
        _drain(fn())
    times = []
    for _ in range(samples):
        _mark_step()
        torch.hpu.synchronize()
        start = time.perf_counter()
        last = None
        for _ in range(inner):
            last = fn()
        _drain(last)
        times.append((time.perf_counter() - start) * 1e3 / inner)
    return statistics.median(times)


def peak_mib(fn) -> tuple[float, float]:
    """(peak allocated MiB during one call, ms of that call)."""
    _drain(fn())
    torch.hpu.synchronize()
    torch.hpu.reset_peak_memory_stats()
    _mark_step()
    start = time.perf_counter()
    out = fn()
    _drain(out)
    elapsed = (time.perf_counter() - start) * 1e3
    return torch.hpu.max_memory_allocated() / 2 ** 20, elapsed


# ---------------------------------------------------------------------------
# baseline (H9(a)) and candidates
# ---------------------------------------------------------------------------
def _baseline(hidden, topk_ids, topk_weights, store):
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped, compact_expert_ids)

    unique, local = compact_expert_ids(topk_ids, store.num_experts)
    gate_w = store.dequant("gate_proj", unique)
    up_w = store.dequant("up_proj", unique)
    down_w = store.dequant("down_proj", unique)
    return apply_routed_experts_grouped(hidden, local, topk_weights, gate_w, up_w, down_w)


def _baseline_parts(hidden, topk_ids, topk_weights, store) -> dict:
    """Split the H9(a) path into the pieces the report has to attribute."""
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        _add_grouped_bucket, apply_routed_experts_grouped, compact_expert_ids,
        materialized_ids)

    devices = hidden.device
    unique, local = compact_expert_ids(topk_ids, store.num_experts)
    num_tokens, top_k = topk_ids.shape
    slots = num_tokens * top_k
    u = int(unique.numel())
    flat_ids = local.reshape(-1).to(dtype=torch.int64).contiguous()
    weights = topk_weights.reshape(-1).contiguous()
    token_of_slot = torch.arange(num_tokens, device=devices).repeat_interleave(top_k)
    experts = torch.arange(u, dtype=torch.int64, device=devices)
    parts = {"unique": u, "slots": slots}

    def compact_only():
        return compact_expert_ids(topk_ids, store.num_experts)

    parts["compact_ms"] = time_sync(compact_only)

    def dequant3():
        return (store.dequant("gate_proj", unique),
                store.dequant("up_proj", unique),
                store.dequant("down_proj", unique))

    gate_w, up_w, down_w = dequant3()
    parts["dequant_ms"] = time_sync(dequant3, samples=2)
    # Hard number, independent of the allocator: the H9(a) path holds all three
    # BF16 projections of every selected expert at the same time.
    parts["bf16_mib"] = (gate_w.numel() + up_w.numel() + down_w.numel()) * 2 / 2 ** 20
    parts["materialized_MiB"] = parts["bf16_mib"]

    def counts_only():
        # The H9(a) path builds this presence matrix TWICE (cap read here, and
        # again inside _add_grouped_bucket): both are timed here.
        counts = (flat_ids.unsqueeze(1) == experts).sum(dim=0)
        cap = min(slots, max(0, int(counts.max())))
        counts2 = (flat_ids.unsqueeze(1) == experts).sum(dim=0)
        return counts + counts2, cap

    _counts_value, cap = counts_only()
    parts["cap"] = int(cap)
    parts["pad_frac"] = 1.0 - slots / (u * cap)
    parts["bucket_rows"] = u * cap
    parts["double_count_matrix_ms"] = time_sync(counts_only)

    def grouped():
        return apply_routed_experts_grouped(hidden, local, topk_weights,
                                            gate_w, up_w, down_w)

    parts["grouped_ms"] = time_sync(grouped, samples=2)

    def add_bucket():
        out = torch.zeros(num_tokens, HIDDEN, dtype=hidden.dtype, device=devices)
        _add_grouped_bucket(out, hidden, flat_ids, weights, token_of_slot,
                            gate_w, up_w, down_w, experts, cap)
        return out

    parts["bucket_add_ms"] = time_sync(add_bucket, samples=2)
    return parts


def _hybrid(hidden, topk_ids, topk_weights, store, k_sparse: int, chunk: int):
    """2-A(i): count-ordered dense chunks + fused-GEMV for the cold experts.

    Experts are count-ordered, so "count <= k_sparse" is a suffix of the
    selection: the dense prefix is ``[0, n_dense)`` and the sparse slots are
    the tail of the count-sorted slot order. Nothing about the arithmetic of a
    slot changes; the cold experts simply never materialize BF16.
    """
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped_chunked, count_ordered_plan, gemv_slots,
        swiglu_gemv_slots)

    num_tokens, top_k = topk_ids.shape
    slots = num_tokens * top_k
    selected, local, plan = count_ordered_plan(topk_ids, store.num_experts, chunk)
    if plan is None:
        return _baseline(hidden, topk_ids, topk_weights, store)
    counts_sorted = plan["counts_sorted"]
    n_dense = int((counts_sorted > k_sparse).sum().item()) if k_sparse > 0 else plan["count"]
    n_dense = min(n_dense, plan["count"])
    # Exact slot split: the dense part owns the slot prefix of the ``n_dense``
    # busiest experts. Truncating the dense chunk's slot range at this offset is
    # what keeps the GEMV side to experts with count <= k_sparse -- the dot
    # products of every slot are unchanged either way.
    sparse_start = int(counts_sorted[:n_dense].sum().item()) if n_dense < plan["count"] else slots
    dense_plan = dict(plan)
    dense_plan["count"] = n_dense
    n_chunks_dense = (n_dense + chunk - 1) // chunk
    dense_plan["starts"] = list(plan["starts"][:n_chunks_dense]) + [sparse_start]
    slot_out = apply_routed_experts_grouped_chunked(
        hidden, local, topk_weights, store, selected, dense_plan, mix=False)
    token_all = torch.arange(num_tokens, device=hidden.device).repeat_interleave(top_k)
    weights_flat = topk_weights.reshape(-1)
    # The GEMV kernel takes at most 512 slots per launch (its S cap).
    for start in range(sparse_start, slots, MAX_GEMV_SLOTS):
        stop = min(start + MAX_GEMV_SLOTS, slots)
        # slots are already in count-sorted order, so a sparse chunk is a slice
        slot_index = plan["slot_order"][start:stop]
        token_c = token_all.index_select(0, slot_index)
        # local rank -> checkpoint expert id for the GEMV index space
        sparse_ids = selected.index_select(
            0, plan["local_sorted"][start:stop]).to(torch.int32)
        x = hidden.index_select(0, token_c)
        act = swiglu_gemv_slots(store, sparse_ids, x)
        if act is None:
            act = F.silu(gemv_slots(store, "gate_proj", sparse_ids, x)) \
                * gemv_slots(store, "up_proj", sparse_ids, x)
        down = gemv_slots(store, "down_proj", sparse_ids, act)
        slot_out[start:stop] = down * weights_flat.index_select(
            0, slot_index).unsqueeze(-1)
    out = torch.zeros(num_tokens, hidden.size(-1), dtype=hidden.dtype,
                      device=hidden.device)
    out.index_add_(0, token_all.index_select(0, plan["slot_order"]), slot_out)
    return out


def _score(out: torch.Tensor | None, reference: torch.Tensor) -> dict:
    # ``prefill_grouped_h18`` declines by design when the step is too small to
    # pay for per-chunk dispatch (``slots < PREFILL_H18_MIN_SLOTS``, 1024: T=64
    # is 640 slots) or in compile-only mode, and the block then runs the H9(a)
    # path. There is nothing to compare in that case -- record the decline
    # instead of dying on ``None.detach()``.
    if out is None:
        return {"declined": True, "cosine": None, "rel_max": None, "rel_rms": None,
                "bf16_bit_diff": None, "bf16_elements": 0}
    # Host-side comparison: a bit-level ``view(torch.uint16)`` is not a lazy
    # HPU op, and every accuracy number here is tiny next to the transfer.
    left = out.detach().float().cpu()
    right = reference.detach().float().cpu()
    gap = (left - right).abs()
    differences = -1
    if out.dtype == torch.bfloat16:
        differences = int((out.detach().cpu().view(torch.uint16)
                           != reference.detach().cpu().view(torch.uint16)).sum().item())
    return {
        "cosine": float(F.cosine_similarity(
            left.flatten().unsqueeze(0), right.flatten().unsqueeze(0)).item()),
        "rel_max": float((gap.max() / right.abs().max().clamp_min(1e-30)).item()),
        "rel_rms": float((gap.pow(2).mean().sqrt()
                          / right.pow(2).mean().sqrt().clamp_min(1e-30)).item()),
        "bf16_bit_diff": differences,
        "bf16_elements": int(out.numel()),
    }


# ---------------------------------------------------------------------------
# phase: parts + candidate sweeps
# ---------------------------------------------------------------------------
def phase_parts(tokens_list, chunks, sparse_ks, seed, log) -> None:
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped_chunked, compact_expert_ids, count_ordered_plan,
        prefill_grouped_h18)

    print("loading layer-0 routed planes ...", flush=True)
    started = time.time()
    store = load_routed_planes()
    torch.hpu.synchronize()
    packed_mib = EXPERTS * (2 * MOE_INTER * HIDDEN // 2 + HIDDEN * MOE_INTER // 2) / 2 ** 20
    print(f"  loaded in {time.time() - started:.1f}s (packed {packed_mib:.0f} MiB + scales)",
          flush=True)

    regimes = log.setdefault("regimes", {})
    for tokens in tokens_list:
        key = f"T={tokens}"
        entry = regimes.setdefault(key, {})
        hidden, topk_ids, topk_weights = real_routing(tokens, seed)
        unique, local = compact_expert_ids(topk_ids, EXPERTS)
        u = int(unique.numel())
        flat = local.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat.cpu(), minlength=u).sort(descending=True).values
        slots = tokens * TOP_K
        entry["routing"] = {
            "slots": slots, "unique": u, "cap_single": int(counts[0].max().item()),
            "count_p50": float(counts.float().median().item()),
            "count_max": int(counts.max().item()), "count_min": int(counts.min().item()),
            "top5_counts": [int(v) for v in counts[:5].tolist()],
            "experts_count_le_4": int((counts <= 4).sum().item()),
            "slots_held_by_count_le_4": int(counts[counts <= 4].sum().item()),
            "single_cap_bucket_rows": u * int(counts.max().item()),
            "single_cap_pad_frac": 1.0 - slots / (u * int(counts.max().item())),
        }
        print(f"\n== {key}: U={u} cap={entry['routing']['cap_single']} "
              f"pad={entry['routing']['single_cap_pad_frac']:.3f} "
              f"rows={entry['routing']['single_cap_bucket_rows']} ==", flush=True)

        # ---- baseline parts -------------------------------------------------
        parts = _baseline_parts(hidden, topk_ids, topk_weights, store)
        baseline_peak, _ = peak_mib(
            lambda: _baseline(hidden, topk_ids, topk_weights, store))
        baseline = {
            "sync_ms": time_sync(lambda: _baseline(hidden, topk_ids, topk_weights, store),
                                 samples=2),
            "batched_ms": time_batched(lambda: _baseline(hidden, topk_ids, topk_weights, store),
                                       inner=3),
            "peak_MiB": baseline_peak,
            "parts": parts,
        }
        reference = _baseline(hidden, topk_ids, topk_weights, store)
        entry["baseline"] = baseline
        print(f"  baseline sync={baseline['sync_ms']:.2f} ms "
              f"batched={baseline['batched_ms']:.2f} ms peak={baseline_peak:.0f} MiB "
              f"(dequant {parts['dequant_ms']:.2f}, grouped {parts['grouped_ms']:.2f}, "
              f"compact {parts['compact_ms']:.2f}, double_count {parts['double_count_matrix_ms']:.2f})",
              flush=True)

        # ---- 2-A(iii) alone: sorted order, single cap (chunk = E) ------------
        def chunked_call(chunk_size):
            sel_c, loc_c, plan_c = count_ordered_plan(topk_ids, EXPERTS, chunk_size)
            return apply_routed_experts_grouped_chunked(
                hidden, loc_c, topk_weights, store, sel_c, plan_c)

        entry["h18_only_dedup"] = {
            "sync_ms": time_sync(lambda: chunked_call(EXPERTS), samples=2),
            "accuracy": _score(chunked_call(EXPERTS), reference),
        }
        print(f"  2-A(iii) dedup-only (chunk=E): {entry['h18_only_dedup']['sync_ms']:.2f} ms "
              f"cos={entry['h18_only_dedup']['accuracy']['cosine']:.8f} "
              f"bitdiff={entry['h18_only_dedup']['accuracy']['bf16_bit_diff']}", flush=True)

        # ---- 2-A(ii): chunk sweep ------------------------------------------
        sweep = []
        best = None
        for chunk in chunks:
            chunk = min(chunk, EXPERTS)
            selected, local_c, plan = count_ordered_plan(topk_ids, EXPERTS, chunk)
            if plan is None or plan["count"] == 0:
                continue
            chunks_n = (plan["count"] + chunk - 1) // chunk
            rows = sum(min(chunk, plan["count"] - i * chunk) * int(plan["caps"][i])
                       for i in range(chunks_n))
            # BF16 weight bytes alive at once: 2 projections for the hot chunk,
            # then 1 for down. This is the hard number; peak_MiB also carries
            # the persistent packed planes (1.17 GiB).
            hot = min(chunk, plan["count"])
            materialized_mib = 2 * hot * BF16_MIB_PER_EXPERT
            out = apply_routed_experts_grouped_chunked(hidden, local_c, topk_weights,
                                                      store, selected, plan)
            score = _score(out, reference)
            entry_peak, entry_ms = peak_mib(lambda: chunked_call(chunk))
            row = {
                "chunk": chunk,
                "chunks": (plan["count"] + chunk - 1) // chunk,
                "bucket_rows": rows,
                "pad_frac": 1.0 - slots / max(rows, 1),
                "sync_ms": time_sync(lambda: chunked_call(chunk), samples=2),
                "batched_ms": time_batched(lambda: chunked_call(chunk), inner=3),
                "peak_MiB": entry_peak,
                "materialized_MiB": materialized_mib,
                "accuracy": score,
            }
            sweep.append(row)
            print(f"  chunk={chunk:4d} chunks={row['chunks']:2d} rows={rows:6d} "
                  f"pad={row['pad_frac']:.3f} sync={row['sync_ms']:6.2f} "
                  f"batched={row['batched_ms']:6.2f} peak={entry_peak:6.0f} MiB "
                  f"cos={score['cosine']:.8f} bitdiff={score['bf16_bit_diff']}", flush=True)
            if best is None or row["sync_ms"] < best["sync_ms"]:
                best = row
        entry["h18_chunk_sweep"] = sweep
        entry["h18_best_chunk"] = best["chunk"] if best else None

        # ---- production entry point (env switch) ----------------------------
        os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H18"] = "1"
        declined = prefill_grouped_h18(hidden, topk_ids, topk_weights, store) is None
        entry["h18_production"] = {
            "declined": declined,
            "sync_ms": 0.0 if declined else time_sync(lambda: prefill_grouped_h18(
                hidden, topk_ids, topk_weights, store), samples=2),
            "batched_ms": 0.0 if declined else time_batched(
                lambda: prefill_grouped_h18(hidden, topk_ids, topk_weights, store),
                inner=3),
            "accuracy": _score(prefill_grouped_h18(hidden, topk_ids, topk_weights, store),
                               reference),
        }
        prod_peak, _ = peak_mib(lambda: prefill_grouped_h18(
            hidden, topk_ids, topk_weights, store))
        entry["h18_production"]["peak_MiB"] = prod_peak
        os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H18"] = "0"
        entry["h18_disabled_sync_ms"] = time_sync(
            lambda: prefill_grouped_h18(hidden, topk_ids, topk_weights, store), samples=2)
        os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H18"] = "1"
        p = entry["h18_production"]
        if declined:
            print(f"  production H18: declined (slots={slots} < "
                  f"PREFILL_H18_MIN_SLOTS) -> the block keeps the H9(a) path",
                  flush=True)
        else:
            print(f"  production H18: sync={p['sync_ms']:.2f} ms batched={p['batched_ms']:.2f} ms "
                  f"peak={prod_peak:.0f} MiB cos={p['accuracy']['cosine']:.8f} "
                  f"bitdiff={p['accuracy']['bf16_bit_diff']}/{p['accuracy']['bf16_elements']} "
                  f"| switch off: prefill_grouped_h18 returns None in "
                  f"{entry['h18_disabled_sync_ms']:.3f} ms", flush=True)

        # ---- 2-A(i): sparse sweep -------------------------------------------
        sparse = []
        chunk_for_sparse = int(entry["h18_best_chunk"] or 64)
        for k_sparse in sparse_ks:
            out = _hybrid(hidden, topk_ids, topk_weights, store, k_sparse, chunk_for_sparse)
            score = _score(out, reference)
            sparse_ms = time_sync(
                lambda: _hybrid(hidden, topk_ids, topk_weights, store, k_sparse,
                                chunk_for_sparse), samples=2)
            n_sparse_experts = int((counts <= k_sparse).sum().item()) if k_sparse else 0
            sparse_row = {
                "k_sparse": k_sparse,
                "sparse_experts": n_sparse_experts,
                "sparse_slots": int(counts[counts <= k_sparse].sum().item()) if k_sparse else 0,
                "dequant_experts_removed": n_sparse_experts,
                "bf16_MiB_removed": n_sparse_experts * BF16_MIB_PER_EXPERT,
                "sync_ms": sparse_ms,
                "accuracy": score,
            }
            sparse.append(sparse_row)
            print(f"  K_sparse={k_sparse}: offload {n_sparse_experts:3d} experts / "
                  f"{sparse_row['sparse_slots']:5d} slots "
                  f"({sparse_row['bf16_MiB_removed']:7.0f} MiB BF16 not written) "
                  f"sync={sparse_ms:6.2f} ms cos={score['cosine']:.8f}", flush=True)
        entry["sparse_sweep"] = sparse
    log["sparse_chunk"] = "best H18 chunk per regime"


# ---------------------------------------------------------------------------
# phase: production Qwen4ExpSparseMoeBlock Before/After
# ---------------------------------------------------------------------------
def _build_block():
    """The production block (same construction as bench_h17_step1_moe_ab.py)."""
    import tempfile

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_gaudi.models.qwen4_exp import Qwen4ExpSparseMoeBlock
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
    dummy = Path(tempfile.mkdtemp()) / "dummy-model"
    dummy.mkdir(parents=True, exist_ok=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
        "num_hidden_layers": 1, "num_key_value_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 262144, "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(model=str(dummy), tokenizer=str(dummy), dtype="bfloat16",
                               seed=0, max_model_len=8192)
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)
    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method=f"file://{tmp}",
                                 local_rank=0, backend="gloo")
    kept = set_current_vllm_config(vllm_config)
    kept.__enter__()
    initialize_model_parallel(1, 1)
    with set_default_torch_dtype(torch.bfloat16):
        block = Qwen4ExpSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")
    return block, cfg


def _block_weights(cfg):
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    out: list[tuple[str, torch.Tensor]] = [
        ("gate.weight", _load(f"{LAYER}.gate.weight").contiguous()),
        ("shared_expert_gate.weight", _load(f"{LAYER}.shared_expert_gate.weight").contiguous()),
    ]
    base = f"{LAYER}.shared_expert.down_proj"
    out.append(("shared_expert.down_proj.weight",
                dequant_mxfp8(_load(base + ".weight"), _load(base + ".weight_scale"))))
    for shard_id, kind in enumerate(("gate_proj", "up_proj")):
        base = f"{LAYER}.shared_expert.{kind}"
        weight = dequant_mxfp8(_load(base + ".weight"), _load(base + ".weight_scale"))
        weight.shard_id = shard_id
        out.append(("shared_expert.gate_up_proj.weight", weight))
    for expert in range(cfg.num_experts):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            base = f"{LAYER}.experts.{expert}.{kind}.weight"
            out.append((f"experts.{expert}.{kind}.weight", _load(base).contiguous()))
            out.append((f"experts.{expert}.{kind}.weight_scale",
                        _load(base + "_scale").view(torch.uint8)))
            out.append((f"experts.{expert}.{kind}.weight_scale_2",
                        _load(base + "_scale_2")))
    return out


def phase_block(tokens_list, seed, log) -> None:
    print("\nbuilding the production Qwen4ExpSparseMoeBlock ...", flush=True)
    started = time.time()
    block, cfg = _build_block()
    block.load_weights(iter(_block_weights(cfg)))
    block = block.to("hpu")
    torch.hpu.synchronize()
    print(f"  ready in {time.time() - started:.0f}s", flush=True)
    section = log.setdefault("block", {})
    for tokens in tokens_list:
        key = f"T={tokens}"
        gen = torch.Generator().manual_seed(seed + tokens)
        hidden = (torch.randn(tokens, HIDDEN, generator=gen) * 0.5).bfloat16().to("hpu")
        os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H18"] = "0"
        before = block(hidden)
        before_peak, _ = peak_mib(lambda: block(hidden))
        before_ms = time_sync(lambda: block(hidden), samples=3)
        os.environ["VLLM_QWEN4EXP_MOE_PREFILL_H18"] = "1"
        after = block(hidden)
        after_peak, _ = peak_mib(lambda: block(hidden))
        after_ms = time_sync(lambda: block(hidden), samples=3)
        after_batched = time_batched(lambda: block(hidden), inner=3)
        score = _score(after, before)
        section[key] = {
            "before_sync_ms": before_ms, "after_sync_ms": after_ms,
            "after_batched_ms": after_batched,
            "before_peak_MiB": before_peak, "after_peak_MiB": after_peak,
            "accuracy": score,
            "checksum_before": float(before.float().sum().item()),
            "checksum_after": float(after.float().sum().item()),
        }
        print(f"  block {key}: before {before_ms:.2f} ms / {before_peak:.0f} MiB -> "
              f"after {after_ms:.2f} ms / {after_peak:.0f} MiB "
              f"({before_ms / after_ms:.2f}x), cos={score['cosine']:.8f} "
              f"rel={score['rel_max']:.3e} bitdiff={score['bf16_bit_diff']}/"
              f"{score['bf16_elements']} checksum {section[key]['checksum_before']:.4f} -> "
              f"{section[key]['checksum_after']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=REPO / "tools/qwen4exp_flashnext/bench_h18_prefill_moe_eager.json")
    parser.add_argument("--tokens", type=str, default="64,128,512,2048")
    parser.add_argument("--chunks", type=str, default="8,16,32,64,128,256,512")
    parser.add_argument("--sparse", type=str, default="0,1,2,3,4,6,8")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--only", choices=("parts", "block", "all"), default="all")
    parser.add_argument("--merge", action="store_true",
                        help="merge into an existing --json instead of overwriting")
    args = parser.parse_args()

    if EXT is None or not KERNEL.is_file():
        raise SystemExit("build the TPC kernels + pytorch extension first")
    torch.ops.load_library(str(EXT))
    if not torch.hpu.is_available():
        raise SystemExit("no HPU")
    tokens_list = [int(t) for t in args.tokens.split(",") if t.strip()]
    chunks = [int(t) for t in args.chunks.split(",") if t.strip()]
    sparse_ks = [int(t) for t in args.sparse.split(",") if t.strip()]

    log: dict = {}
    if args.merge and args.json.is_file():
        log = json.loads(args.json.read_text())
    log.update({
        "unit": "H18 Step 2 / Part 2 (Direction 2-A: prefill MoE HBM round trip)",
        "checkpoint": str(CHECKPOINT), "layer": LAYER,
        "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "geometry": {"hidden": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K,
                     "moe_inter": MOE_INTER,
                     "bf16_MiB_per_expert_all_3_proj": BF16_MIB_PER_EXPERT},
        "tokens": tokens_list,
    })
    if args.only in ("parts", "all"):
        phase_parts(tokens_list, chunks, sparse_ks, args.seed, log)
    if args.only in ("block", "all"):
        phase_block(tokens_list, args.seed, log)

    # Before -> After summary, computed from the numbers.
    summary = {}
    for key, entry in (log.get("regimes") or {}).items():
        base = entry.get("baseline") or {}
        prod = entry.get("h18_production") or {}
        if base and prod and prod.get("sync_ms"):
            summary[key] = {
                "before_ms": base["sync_ms"], "after_ms": prod["sync_ms"],
                "speedup": base["sync_ms"] / prod["sync_ms"] if prod["sync_ms"] else None,
                "delta_pct": 100.0 * (prod["sync_ms"] - base["sync_ms"]) / base["sync_ms"],
                "before_peak_MiB": base["peak_MiB"], "after_peak_MiB": prod["peak_MiB"],
                "peak_reduction_pct": 100.0 * (1 - prod["peak_MiB"] / base["peak_MiB"]
                                               ) if base["peak_MiB"] else None,
                "cosine": prod["accuracy"]["cosine"],
                "bf16_bit_diff": prod["accuracy"]["bf16_bit_diff"],
                "bf16_elements": prod["accuracy"]["bf16_elements"],
            }
    if summary:
        log["before_after"] = summary
    args.json.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    if summary:
        print("Before -> After (routed MoE, sync ms / peak MiB):")
        for key, row in summary.items():
            print(f"  {key}: {row['before_ms']:.2f} -> {row['after_ms']:.2f} ms "
                  f"({row['delta_pct']:+.1f}%, {row['speedup']:.2f}x) | "
                  f"{row['before_peak_MiB']:.0f} -> {row['after_peak_MiB']:.0f} MiB "
                  f"(-{row['peak_reduction_pct']:.1f}%) | cos={row['cosine']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
