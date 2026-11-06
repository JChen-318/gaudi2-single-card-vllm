# SPDX-License-Identifier: Apache-2.0
"""H22 Phase 1 — prefill MoE: consolidated 3-launch projection-first dequant,
single bucket gather, tail-expert GEMV crossover, Layer-0 vs Layer-3 histograms.

Real Layer-0 and Layer-3 routed planes (``~/models/qwen4exp-flashnext-plain``)
and the real router of each layer. E=512, top-10, H=2560, I=640, T in {512, 2048}.

What is measured, per (layer, T)
--------------------------------
1. ``histogram`` — the exact per-expert slot-count distribution of that layer's
   router: how many experts have count == 1, == 2, ..., <= K, and how many slots
   those experts hold. This is the quantity that decides whether an expert is
   worth dequantizing (dequant cost is per EXPERT, bmm cost is per SLOT).
2. ``decompose`` — the H18 loop taken apart, all on the same plan:
     (a) ``count_ordered_plan`` (presence matrix + sort + the one host read)
     (b) 24 x dequant (8 chunks x 3 projections, what H18 does today) vs
         3 x dequant (one launch per projection over all U_dense experts)
     (c) per-chunk bucket construction (8 x new_zeros + index_copy + view) vs
         the single pre-built bucket (one cell vector + one index_copy)
     (d) per-chunk tight-cap bmm vs the single-cap bmm; then the
         ``slot_out`` gather + ``index_add_`` mix
3. ``candidates`` — end-to-end routed MoE, bit-compared against the H9(a)
   baseline: H18 (production), 2-A with n_super projection groups, the
   ``empty``-bucket variant, and 2-B (tail experts on the fused GEMV).
4. ``crossover`` — cost per dequantized expert vs cost per GEMV slot at this
   exact U, i.e. the count at which the GEMV stops being cheaper than
   dequant + bmm, plus the multi-token-GEMV upper bound (a kernel that serves
   ``ceil(count / Mtok)`` weight passes per expert instead of ``count``).

Nothing is served and no profiler runs. Eager mode (``PT_HPU_LAZY_MODE=0``);
lazy shares the same Python glue, and the H18/H21 records cover its ratios.

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h22_prefill_moe.py \
      --json tools/qwen4exp_flashnext/bench_h22_prefill_moe_eager.json
"""
from __future__ import annotations

import argparse
import json
import math
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
GEMV_MAX_SLOTS = 512  # swiglu GEMV kernel S cap
# gate [I,H] + up [I,H] + down [H,I] BF16 = 9.375 MiB per expert, 3.125 per projection
BF16_MIB_PER_PROJ = (MOE_INTER * HIDDEN * 2) / 2 ** 20
BF16_MIB_PER_EXPERT = 3 * BF16_MIB_PER_PROJ

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


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp"


def load_routed_planes(layer: int, device: str = "hpu"):
    """The three packed routed-expert planes of one layer, on the device."""
    from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts

    base_prefix = layer_prefix(layer)
    store = PackedRoutedExperts(EXPERTS)
    for kind, (n, k) in (("gate_proj", (MOE_INTER, HIDDEN)),
                         ("up_proj", (MOE_INTER, HIDDEN)),
                         ("down_proj", (HIDDEN, MOE_INTER))):
        packed = torch.empty(EXPERTS, n, k // 2, dtype=torch.uint8)
        scales = torch.empty(EXPERTS, n, k // 16, dtype=torch.uint8)
        scale2 = torch.empty(EXPERTS, dtype=torch.float32)
        for expert in range(EXPERTS):
            base = f"{base_prefix}.experts.{expert}.{kind}"
            packed[expert] = _load(base + ".weight").view(torch.uint8).reshape(n, k // 2)
            scales[expert] = _load(base + ".weight_scale").view(torch.uint8).reshape(n, k // 16)
            scale2[expert] = float(_load(base + ".weight_scale_2").reshape(-1)[0])
        setattr(store, f"{kind}_packed", packed.to(device))
        setattr(store, f"{kind}_scale", scales.to(device))
        setattr(store, f"{kind}_scale2", scale2.to(device))
    return store


def real_routing(layer: int, tokens: int, seed: int = 20261105):
    """That layer's router on a seeded hidden state (the production router GEMM)."""
    gate_weight = _load(f"{layer_prefix(layer)}.gate.weight").to("hpu").bfloat16()
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
    if isinstance(out, dict):
        return sum(_drain(item) for item in out.values() if isinstance(item, torch.Tensor))
    _mark_step()
    torch.hpu.synchronize()
    return float(out.reshape(-1)[0].float().item()) if out.numel() else 0.0


def time_sync(fn, warm: int = 1, samples: int = 3) -> float:
    """Median ms with a synchronize around every call (launch attribution)."""
    check = 0.0
    for _ in range(warm):
        check = _drain(fn())
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


def _score(out: torch.Tensor, reference: torch.Tensor) -> dict:
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
# H22 shape plan helpers: one concatenated bucket for every chunk
# ---------------------------------------------------------------------------
def chunk_geometry(count: int, chunk: int, caps: list[int]) -> tuple[int, list[int], list[int]]:
    """(n_chunks, rows_per_chunk, bucket_row_base) for a count-ordered plan."""
    n_chunks = (count + chunk - 1) // chunk
    rows = [min(chunk, count - index * chunk) * int(caps[index]) for index in range(n_chunks)]
    bases = [0]
    for row in rows:
        bases.append(bases[-1] + row)
    return n_chunks, rows, bases


def bucket_cells(plan: dict, chunk: int, dense_slots: int, bases: list[int],
                 device) -> tuple[torch.Tensor, torch.Tensor]:
    """Global and per-chunk bucket cells for the first ``dense_slots`` slots.

    Both come from one vectorized expression over the whole dense range, so the
    layer pays ONE cell construction instead of one per chunk. ``cell_local``
    indexes a chunk's own ``[n_c * cap_c]`` flattened projection (the gather),
    ``cell`` indexes the single concatenated bucket.
    """
    rank = plan["local_sorted"][:dense_slots]
    caps_t = torch.tensor([int(v) for v in plan["caps"][:len(bases) - 1]],
                          dtype=torch.int64, device=device)
    base_t = torch.tensor(bases[:-1], dtype=torch.int64, device=device)
    which = rank // chunk
    cap_v = caps_t.index_select(0, which)
    base_v = base_t.index_select(0, which)
    cell_local = (rank - which * chunk) * cap_v + plan["rank_sorted"][:dense_slots]
    return cell_local + base_v, cell_local


def _sorted_inputs(hidden, topk_ids, topk_weights, plan):
    num_tokens, top_k = topk_ids.shape
    device = hidden.device
    token_sorted = torch.arange(num_tokens, device=device).repeat_interleave(top_k) \
        .index_select(0, plan["slot_order"])
    weight_sorted = topk_weights.reshape(-1).index_select(0, plan["slot_order"])
    x_sorted = hidden.index_select(0, token_sorted)
    return token_sorted, weight_sorted, x_sorted


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def routed_h22(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store,
    plan: dict,
    *,
    chunk: int = 64,
    n_super: int = 1,
    bucket_empty: bool = True,
    tail_k: int = 0,
    tail_mode: str = "swiglu",
    mix: bool = True,
):
    """H22 2-A + optional 2-B.

    Projection-by-projection over ``n_super`` expert groups: gate is dequantized
    for a whole group, every chunk's tight-cap bmm runs, the BF16 gate rows die,
    then up, then down. ``n_super=1`` is 3 dequant launches per layer (one
    projection live at a time); ``n_super=8`` reproduces H18's 24 launches but
    with only ONE projection resident instead of two.

    The bucket is a single allocation for every chunk (one cell vector, one
    ``index_copy_``); each chunk still bmms against its own ``cap_c``.

    ``tail_k > 0`` sends the experts with ``1 <= count <= tail_k`` to the fused
    NVFP4 GEMV instead of dequantizing them (not bit-exact: the GEMV dots in
    F32 lanes, the MME bmm in its own order).
    """
    num_tokens, top_k = topk_ids.shape
    hidden_size = hidden.shape[-1]
    device = hidden.device
    slots = num_tokens * top_k
    count = int(plan["count"])
    token_sorted, weight_sorted, x_sorted = _sorted_inputs(hidden, topk_ids, topk_weights, plan)

    # ---- 2-B: the tail is a suffix of the count-ordered selection ----------
    n_dense = count
    dense_slots = slots
    if tail_k > 0:
        counts_sorted = plan["counts_sorted"]
        n_dense = int((counts_sorted > tail_k).sum().item())
        n_dense = max(0, min(n_dense, count))
        dense_slots = int(counts_sorted[:n_dense].sum().item()) if n_dense < count else slots
    # Geometry of the DENSE range only. The last dense chunk can be truncated
    # by the 2-B tail, and its cap is unchanged by that truncation: every dense
    # expert holds more slots than every tail expert, so extending the range
    # cannot raise the chunk's max count.
    n_chunks, rows, bases = chunk_geometry(n_dense, chunk, plan["caps"])

    slot_out = torch.empty(slots, hidden_size, dtype=hidden.dtype, device=device)
    if n_dense > 0:
        cell, cell_local = bucket_cells(plan, chunk, dense_slots, bases, device)
        total_rows = bases[-1]
        bucket = torch.empty(total_rows, hidden_size, dtype=hidden.dtype, device=device) \
            if bucket_empty else hidden.new_zeros(total_rows, hidden_size)
        bucket.index_copy_(0, cell, x_sorted[:dense_slots])
        caps = [int(v) for v in plan["caps"][:n_chunks]]
        # The last boundary is the dense slot count, not prefix[n_chunks*chunk]:
        # with a 2-B tail the chunk grid runs past ``n_dense``, whose experts
        # still hold slots that belong to the tail.
        starts = [int(v) for v in plan["starts"][:n_chunks]] + [dense_slots]

        chunks_per_super = max(1, math.ceil(n_chunks / max(1, n_super)))
        supers = [(lo, min(n_chunks, lo + chunks_per_super))
                  for lo in range(0, n_chunks, chunks_per_super)]

        # ---- phase 1: gate ------------------------------------------------
        gates: list[torch.Tensor] = [None] * n_chunks  # type: ignore[list-item]
        for lo, hi in supers:
            ids_sc = _ids(store, plan, lo * chunk, min(n_dense, hi * chunk))
            gate_w = store.dequant("gate_proj", ids_sc)
            for index in range(lo, hi):
                n_c = min(chunk, n_dense - index * chunk)
                cap_c = caps[index]
                rows_c = rows[index]
                x_c = bucket[bases[index]:bases[index] + rows_c].view(n_c, cap_c, hidden_size)
                w_c = gate_w[(index - lo) * chunk:(index - lo) * chunk + n_c]
                gates[index] = torch.bmm(x_c, w_c.transpose(1, 2))
            del gate_w

        # ---- phase 2: up + SwiGLU -----------------------------------------
        activated: list[torch.Tensor] = [None] * n_chunks  # type: ignore[list-item]
        for lo, hi in supers:
            ids_sc = _ids(store, plan, lo * chunk, min(n_dense, hi * chunk))
            up_w = store.dequant("up_proj", ids_sc)
            for index in range(lo, hi):
                n_c = min(chunk, n_dense - index * chunk)
                cap_c = caps[index]
                rows_c = rows[index]
                x_c = bucket[bases[index]:bases[index] + rows_c].view(n_c, cap_c, hidden_size)
                w_c = up_w[(index - lo) * chunk:(index - lo) * chunk + n_c]
                up_out = torch.bmm(x_c, w_c.transpose(1, 2))
                activated[index] = F.silu(gates[index]) * up_out
                gates[index] = None
                del up_out
            del up_w

        # ---- phase 3: down + weighted gather ------------------------------
        for lo, hi in supers:
            ids_sc = _ids(store, plan, lo * chunk, min(n_dense, hi * chunk))
            down_w = store.dequant("down_proj", ids_sc)
            for index in range(lo, hi):
                n_c = min(chunk, n_dense - index * chunk)
                rows_c = rows[index]
                w_c = down_w[(index - lo) * chunk:(index - lo) * chunk + n_c]
                projected = torch.bmm(activated[index], w_c.transpose(1, 2))
                activated[index] = None
                stop = starts[index + 1]
                picked = projected.view(rows_c, hidden_size) \
                    .index_select(0, cell_local[starts[index]:stop])
                slot_out[starts[index]:stop] = picked * weight_sorted[starts[index]:stop].unsqueeze(-1)
                del projected
            del down_w
        del bucket

    # ---- 2-B tail: fused NVFP4 GEMV on 1 <= count <= tail_k ---------------
    if tail_k > 0 and dense_slots < slots:
        from vllm_gaudi.models.qwen4exp.packed_moe import gemv_slots, swiglu_gemv_slots

        token_all = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
        weights_flat = topk_weights.reshape(-1)
        for start in range(dense_slots, slots, GEMV_MAX_SLOTS):
            stop = min(start + GEMV_MAX_SLOTS, slots)
            slot_index = plan["slot_order"][start:stop]
            token_c = token_all.index_select(0, slot_index)
            tail_ids = plan["selected_by_rank"].index_select(
                0, plan["local_sorted"][start:stop]).to(torch.int32)
            x = hidden.index_select(0, token_c)
            act = swiglu_gemv_slots(store, tail_ids, x) if tail_mode == "swiglu" else None
            if act is None:
                act = F.silu(gemv_slots(store, "gate_proj", tail_ids, x)) \
                    * gemv_slots(store, "up_proj", tail_ids, x)
            down = gemv_slots(store, "down_proj", tail_ids, act)
            slot_out[start:stop] = down * weights_flat.index_select(
                0, slot_index).unsqueeze(-1)
    if not mix:
        return slot_out
    out = torch.zeros(num_tokens, hidden_size, dtype=hidden.dtype, device=device)
    out.index_add_(0, token_sorted, slot_out)
    return out


def _ids(store, plan: dict, lo: int, hi: int) -> torch.Tensor:
    """Materialized expert-id slice for expert ranks ``[lo, hi)``."""
    from vllm_gaudi.models.qwen4exp.packed_moe import materialized_ids

    return materialized_ids(plan["selected_by_rank"][lo:hi], plan["selected_by_rank"].device)


def routed_h18(hidden, topk_ids, topk_weights, store, plan: dict):
    """The shipped H18 body, plan pre-computed (timed without the plan)."""
    from vllm_gaudi.models.qwen4exp.packed_moe import apply_routed_experts_grouped_chunked

    return apply_routed_experts_grouped_chunked(
        hidden, plan["local"], topk_weights, store, plan["selected_by_rank"], plan)


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def histogram(counts: torch.Tensor) -> dict:
    """Exact count histogram over the present experts (descending order kept)."""
    values = [int(v) for v in counts.tolist() if int(v) > 0]
    total_slots = sum(values)
    entry: dict = {
        "unique": len(values),
        "slots": total_slots,
        "count_max": max(values) if values else 0,
        "count_min": min(values) if values else 0,
        "count_p50": statistics.median(values) if values else 0,
        "exact": {str(k): sum(1 for v in values if v == k) for k in range(1, 9)},
        "slots_exact": {str(k): sum(v for v in values if v == k) for k in range(1, 9)},
        "cumulative": {}, "slots_cumulative": {},
    }
    for k in (1, 2, 3, 4, 6, 8, 16, 32):
        entry["cumulative"][str(k)] = sum(1 for v in values if v <= k)
        entry["slots_cumulative"][str(k)] = sum(v for v in values if v <= k)
    entry["gt_8"] = sum(1 for v in values if v > 8)
    entry["slots_gt_8"] = sum(v for v in values if v > 8)
    return entry


def _plan_for(topk_ids, chunk: int, store) -> dict:
    from vllm_gaudi.models.qwen4exp.packed_moe import count_ordered_plan

    selected, local, plan = count_ordered_plan(topk_ids, store.num_experts, chunk)
    plan = dict(plan)
    plan["selected_by_rank"] = selected
    plan["local"] = local
    return plan


def phase_decompose(hidden, topk_ids, topk_weights, store, plan, chunk, entry, log) -> None:
    """(a)-(d): every piece of the H18 loop, timed on the same plan."""
    from vllm_gaudi.models.qwen4exp.packed_moe import materialized_ids

    device = hidden.device
    slots = hidden.shape[0] * TOP_K
    count = int(plan["count"])
    caps = [int(v) for v in plan["caps"]]
    starts = [int(v) for v in plan["starts"]]
    n_chunks, rows, bases = chunk_geometry(count, chunk, caps)
    token_sorted, weight_sorted, x_sorted = _sorted_inputs(hidden, topk_ids, topk_weights, plan)
    selected = plan["selected_by_rank"]
    cell, cell_local = bucket_cells(plan, chunk, slots, bases, device)
    total_rows = bases[-1]
    out: dict = {
        "unique": count, "chunks": n_chunks,
        "bucket_rows_single_cap": count * caps[0],
        "bucket_rows_chunked": total_rows,
        "pad_frac_single_cap": 1.0 - slots / max(count * caps[0], 1),
        "pad_frac_chunked": 1.0 - slots / max(total_rows, 1),
        "caps": caps[:n_chunks],
    }

    # (a) the plan ------------------------------------------------------
    from vllm_gaudi.models.qwen4exp.packed_moe import count_ordered_plan

    out["plan_ms"] = time_sync(
        lambda: count_ordered_plan(topk_ids, store.num_experts, chunk), samples=3)
    out["plan_ms_batched"] = time_batched(
        lambda: count_ordered_plan(topk_ids, store.num_experts, chunk), inner=4)

    # (a2) what the plan is made of -------------------------------------
    flat = topk_ids.reshape(-1).to(torch.int64)
    experts = torch.arange(store.num_experts, dtype=torch.int64, device=device)
    ones = torch.ones(flat.numel(), dtype=torch.int32, device=device)
    counts_ref = plan["counts_sorted"]

    def counts_matrix():
        return (flat.unsqueeze(1) == experts).sum(dim=0).to(torch.int32)

    def counts_scatter():
        out_c = torch.zeros(store.num_experts, dtype=torch.int32, device=device)
        out_c.scatter_add_(0, flat, ones)
        return out_c

    def sort_counts():
        return torch.sort(counts_ref, descending=True, stable=True)

    def slot_sort():
        return torch.argsort(plan["local"].reshape(-1).to(torch.int64), stable=True)

    diff = int((counts_matrix().float() - counts_scatter().float()).abs().sum().item())
    out["counts_matrix_equals_scatter"] = diff == 0
    out["counts_matrix_ms"] = time_sync(counts_matrix, samples=3)
    out["counts_scatter_ms"] = time_sync(counts_scatter, samples=3)
    out["sort_counts_ms"] = time_sync(sort_counts, samples=3)
    out["slot_argsort_ms"] = time_sync(slot_sort, samples=3)
    out["rank_gather_ms"] = time_sync(
        lambda: torch.zeros(store.num_experts, dtype=torch.int32, device=device)
        .index_copy_(0, selected.to(torch.int64),
                     torch.arange(count, dtype=torch.int32, device=device))
        .index_select(0, flat), samples=3)

    # (b) dequant launches ----------------------------------------------
    ids_all = materialized_ids(selected, device)

    def dequant_3():
        return (store.dequant("gate_proj", ids_all),
                store.dequant("up_proj", ids_all),
                store.dequant("down_proj", ids_all))

    chunk_ids = [materialized_ids(selected[i * chunk:min(count, (i + 1) * chunk)], device)
                 for i in range(n_chunks)]

    def dequant_by_proj():
        # 2-A with n_super = n_chunks: one projection live, 8 launches each.
        # The outputs are RETURNED: a call that returns None measures only the
        # async dispatch (0.5 ms for 24 launches) and not the unpack itself.
        kept = []
        for kind in ("gate_proj", "up_proj", "down_proj"):
            for piece in chunk_ids:
                kept.append(store.dequant(kind, piece))
        return kept

    def dequant_h18():
        # exactly H18's loop body order: 3 launches per chunk
        kept = []
        for piece in chunk_ids:
            kept.append(store.dequant("gate_proj", piece))
            kept.append(store.dequant("up_proj", piece))
            kept.append(store.dequant("down_proj", piece))
        return kept

    out["dequant_3launch_ms"] = time_sync(dequant_3, samples=2)
    out["dequant_by_proj_24launch_ms"] = time_sync(dequant_by_proj, samples=2)
    out["dequant_h18_24launch_ms"] = time_sync(dequant_h18, samples=2)
    peak3, _ = peak_mib(dequant_3)
    peak_chunk, _ = peak_mib(lambda: [store.dequant("gate_proj", piece) for piece in chunk_ids])
    out["dequant_3launch_peak_MiB"] = peak3
    out["dequant_onechunk_peak_MiB"] = peak_chunk
    out["bf16_MiB_all_experts_one_proj"] = count * BF16_MIB_PER_PROJ
    chain = dequant_3()
    torch.hpu.synchronize()
    out["dequant_3launch_bf16_MiB"] = sum(t.numel() for t in chain) * 2 / 2 ** 20
    del chain

    # (c) bucket construction -------------------------------------------
    def bucket_chunked():
        for index in range(n_chunks):
            stop = starts[index + 1]
            cap_c = caps[index]
            n_c = min(chunk, count - index * chunk)
            local_cell = cell_local[starts[index]:stop] - bases[index]
            piece = hidden.new_zeros(n_c * cap_c, HIDDEN)
            piece.index_copy_(0, local_cell, x_sorted[starts[index]:stop])
            piece.view(n_c, cap_c, HIDDEN)
        return piece

    def bucket_single_zeros():
        piece = hidden.new_zeros(total_rows, HIDDEN)
        piece.index_copy_(0, cell, x_sorted[:slots])
        return piece

    def bucket_single_empty():
        piece = torch.empty(total_rows, HIDDEN, dtype=hidden.dtype, device=device)
        piece.index_copy_(0, cell, x_sorted[:slots])
        return piece

    out["bucket_chunked_ms"] = time_sync(bucket_chunked, samples=3)
    out["bucket_single_zeros_ms"] = time_sync(bucket_single_zeros, samples=3)
    out["bucket_single_empty_ms"] = time_sync(bucket_single_empty, samples=3)
    out["bucket_pad_bytes_MiB"] = total_rows * HIDDEN * 2 / 2 ** 20

    # (d) bmm: per-chunk tight cap vs single cap, plus the mix ------------
    gate_all = store.dequant("gate_proj", ids_all)
    up_all = store.dequant("up_proj", ids_all)
    down_all = store.dequant("down_proj", ids_all)
    gate_pieces = [materialized_ids(selected[i * chunk:min(count, (i + 1) * chunk)], device)
                   for i in range(n_chunks)]
    gate_w_c = [store.dequant("gate_proj", piece) for piece in gate_pieces]
    down_w_c = [store.dequant("down_proj", piece) for piece in gate_pieces]
    bucket = bucket_single_zeros()
    bucket_empty = torch.empty_like(bucket)
    bucket_empty.copy_(bucket)
    single_cap = caps[0]
    single_bucket = torch.zeros(count * single_cap, HIDDEN, dtype=hidden.dtype, device=device)
    ranks_all = plan["local_sorted"][:slots]
    single_cell = ranks_all * single_cap + plan["rank_sorted"][:slots]
    single_bucket.index_copy_(0, single_cell, x_sorted)
    torch.hpu.synchronize()

    def bmm_chunked():
        last = None
        for index in range(n_chunks):
            n_c = min(chunk, count - index * chunk)
            cap_c = caps[index]
            rows_c = rows[index]
            x_c = bucket[bases[index]:bases[index] + rows_c].view(n_c, cap_c, HIDDEN)
            gate_out = torch.bmm(x_c, gate_w_c[index].transpose(1, 2))
            del gate_out
            last = torch.bmm(x_c, gate_w_c[index].transpose(1, 2))
        return last

    def bmm_single_cap():
        return torch.bmm(single_bucket.view(count, single_cap, HIDDEN),
                         gate_all.transpose(1, 2))

    def bmm_chunked_down():
        last = None
        act = bucket[:, :MOE_INTER].contiguous()
        for index in range(n_chunks):
            n_c = min(chunk, count - index * chunk)
            cap_c = caps[index]
            rows_c = rows[index]
            a_c = act[bases[index]:bases[index] + rows_c].view(n_c, cap_c, MOE_INTER)
            last = torch.bmm(a_c, down_w_c[index].transpose(1, 2))
        return last

    def bmm_single_down():
        act = torch.zeros(count * single_cap, MOE_INTER, dtype=hidden.dtype, device=device)
        return torch.bmm(act.view(count, single_cap, MOE_INTER), down_all.transpose(1, 2))

    out["bmm_gate_chunked_ms"] = time_sync(bmm_chunked, samples=2)
    out["bmm_gate_single_cap_ms"] = time_sync(bmm_single_cap, samples=2)
    out["bmm_down_chunked_ms"] = time_sync(bmm_chunked_down, samples=2)
    out["bmm_down_single_cap_ms"] = time_sync(bmm_single_down, samples=2)

    # (d) the mix: gather the projected rows, weight them, store, index_add_
    projected = [torch.zeros(rows[i], HIDDEN, dtype=hidden.dtype, device=device)
                 for i in range(n_chunks)]
    slot_out = torch.empty(slots, HIDDEN, dtype=hidden.dtype, device=device)

    def mix_pieces():
        for index in range(n_chunks):
            stop = starts[index + 1]
            local_cell = cell_local[starts[index]:stop] - bases[index]
            picked = projected[index].index_select(0, local_cell)
            slot_out[starts[index]:stop] = picked * weight_sorted[starts[index]:stop].unsqueeze(-1)
        return slot_out

    def mix_index_add():
        out_buf = torch.zeros(hidden.shape[0], HIDDEN, dtype=hidden.dtype, device=device)
        out_buf.index_add_(0, token_sorted, slot_out)
        return out_buf

    out["mix_gather_ms"] = time_sync(mix_pieces, samples=2)
    out["mix_index_add_ms"] = time_sync(mix_index_add, samples=2)
    out["mix_index_add_batched_ms"] = time_batched(mix_index_add, inner=4)
    del gate_all, up_all, down_all, gate_w_c, down_w_c, bucket, bucket_empty
    del single_bucket, projected, slot_out
    entry["decompose"] = out
    print(f"  (a) plan {out['plan_ms']:.2f} ms (batched {out['plan_ms_batched']:.2f}): "
          f"counts matrix {out['counts_matrix_ms']:.3f} / scatter {out['counts_scatter_ms']:.3f} "
          f"(equal={out['counts_matrix_equals_scatter']}), sort counts "
          f"{out['sort_counts_ms']:.3f}, slot argsort {out['slot_argsort_ms']:.3f}, "
          f"rank gather {out['rank_gather_ms']:.3f} | ", flush=True)
    print(f"  (b) dequant 3x launch "
          f"(b) dequant 3x {out['dequant_3launch_ms']:.2f} / 24x "
          f"{out['dequant_by_proj_24launch_ms']:.2f} (H18 order "
          f"{out['dequant_h18_24launch_ms']:.2f}) ms, peak "
          f"{out['dequant_3launch_peak_MiB']:.0f} / {out['dequant_onechunk_peak_MiB']:.0f} MiB", flush=True)
    print(f"  (c) bucket chunked {out['bucket_chunked_ms']:.2f} / single-zeros "
          f"{out['bucket_single_zeros_ms']:.2f} / single-empty "
          f"{out['bucket_single_empty_ms']:.2f} ms, rows {out['bucket_rows_chunked']} vs "
          f"{out['bucket_rows_single_cap']} (pad {out['pad_frac_chunked']:.3f} vs "
          f"{out['pad_frac_single_cap']:.3f})", flush=True)
    print(f"  (d) bmm gate chunked {out['bmm_gate_chunked_ms']:.2f} / single-cap "
          f"{out['bmm_gate_single_cap_ms']:.2f} | down {out['bmm_down_chunked_ms']:.2f} / "
          f"{out['bmm_down_single_cap_ms']:.2f} | mix gather {out['mix_gather_ms']:.2f} + "
          f"index_add {out['mix_index_add_ms']:.2f} ms", flush=True)


def phase_candidates(hidden, topk_ids, topk_weights, store, reference, entry, chunk,
                     supers, tails, log) -> None:
    """End-to-end routed MoE for every candidate, bit-compared to H9(a)."""
    from vllm_gaudi.models.qwen4exp.packed_moe import prefill_grouped_h18

    plan = _plan_for(topk_ids, chunk, store)
    rows: dict = {}

    def record(name, fn):
        out = fn()
        score = _score(out, reference)
        peak, _ = peak_mib(fn)
        rows[name] = {
            "sync_ms": time_sync(fn, samples=2),
            "batched_ms": time_batched(fn, inner=3),
            "peak_MiB": peak,
            "accuracy": score,
        }
        del out
        row = rows[name]
        print(f"  {name:28s} sync={row['sync_ms']:6.2f} batched={row['batched_ms']:6.2f} "
              f"peak={peak:6.0f} MiB cos={score['cosine']:.8f} "
              f"bitdiff={score['bf16_bit_diff']}/{score['bf16_elements']}", flush=True)

    record("h18_production", lambda: prefill_grouped_h18(
        hidden, topk_ids, topk_weights, store))
    # End-to-end: own plan included, so these are directly comparable with
    # h18_production (which also pays its plan).
    for n_super in supers:
        record(f"h22_full_super{n_super}", lambda ns=n_super: routed_h22(
            hidden, topk_ids, topk_weights, store, _plan_for(topk_ids, chunk, store),
            chunk=chunk, n_super=ns, bucket_empty=True))
    for n_super in supers:
        for empty in (True, False):
            tag = "empty" if empty else "zeros"
            record(f"h22_2a_super{n_super}_{tag}", lambda ns=n_super, e=empty: routed_h22(
                hidden, topk_ids, topk_weights, store, plan,
                chunk=chunk, n_super=ns, bucket_empty=e))
    for tail_k in tails:
        if tail_k <= 0:
            continue
        for mode in ("swiglu", "twogemv"):
            record(f"h22_2b_tail{tail_k}_{mode}", lambda tk=tail_k, m=mode: routed_h22(
                hidden, topk_ids, topk_weights, store, plan,
                chunk=chunk, n_super=1, bucket_empty=True, tail_k=tk, tail_mode=m))
    entry["candidates"] = rows


def phase_crossover(hidden, topk_ids, topk_weights, store, plan, entry, tails) -> None:
    """Cost per dequantized expert vs cost per GEMV slot at this U."""
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        gemv_slots, materialized_ids, swiglu_gemv_slots)

    device = hidden.device
    count = int(plan["count"])
    slots = hidden.shape[0] * TOP_K
    selected = plan["selected_by_rank"]
    out: dict = {"unique": count, "slots": slots}

    # per-expert dequant cost at this U (marginal: U vs U - n_tail experts)
    def dequant_n(n_experts):
        ids = materialized_ids(selected[:n_experts], device)
        return [store.dequant(kind, ids) for kind in ("gate_proj", "up_proj", "down_proj")]

    out["dequant_all3_ms"] = time_sync(lambda: dequant_n(count), samples=2)
    out["dequant_ms_per_expert"] = out["dequant_all3_ms"] / max(count, 1)

    # per-slot GEMV cost for the 3 projections at the tail's S
    token_all = torch.arange(hidden.shape[0], device=device).repeat_interleave(TOP_K)
    weights_flat = topk_weights.reshape(-1)
    counts_sorted = plan["counts_sorted"]

    def gemv_tail(n_tail: int, mode: str, mtok: int = 1):
        n_dense = count - n_tail
        start = int(counts_sorted[:n_dense].sum().item()) if n_dense else 0
        stop = int(counts_sorted.sum().item())
        ranks = plan["local_sorted"][start:stop]
        slot_index = plan["slot_order"][start:stop]
        if mtok > 1:
            # one id per ceil(count / mtok) weight pass: the upper bound a
            # multi-token GEMV would pay (same weight reads, M tokens served).
            per_expert = counts_sorted[n_dense:count].tolist()
            ids_list = []
            for offset, value in enumerate(per_expert):
                ids_list.extend([n_dense + offset] * max(1, math.ceil(value / mtok)))
            sel = selected.index_select(
                0, torch.tensor(ids_list, dtype=torch.int64, device=device)).to(torch.int32)
            x = hidden.index_select(0, torch.zeros(len(ids_list), dtype=torch.int64,
                                                   device=device))
        else:
            sel = selected.index_select(0, ranks).to(torch.int32)
            x = hidden.index_select(0, token_all.index_select(0, slot_index))
        if mode == "swiglu":
            act = swiglu_gemv_slots(store, sel, x)
        else:
            act = F.silu(gemv_slots(store, "gate_proj", sel, x)) \
                * gemv_slots(store, "up_proj", sel, x)
        down = gemv_slots(store, "down_proj", sel, act)
        return down * weights_flat.index_select(
            0, slot_index if mtok == 1 else torch.arange(len(ids_list), device=device)).unsqueeze(-1)

    crossover_rows = []
    for tail_k in tails:
        if tail_k <= 0:
            continue
        n_tail = int((counts_sorted[:count] <= tail_k).sum().item())
        tail_slots = int(counts_sorted[count - n_tail:count].sum().item()) if n_tail else 0
        row = {"tail_k": tail_k, "tail_experts": n_tail, "tail_slots": tail_slots}
        if n_tail and tail_slots:
            for mode in ("swiglu", "twogemv"):
                row[f"gemv_{mode}_ms"] = time_sync(
                    lambda tk=tail_k, m=mode: gemv_tail(tk, m), samples=2)
                row[f"gemv_{mode}_us_per_slot"] = \
                    1e3 * row[f"gemv_{mode}_ms"] / max(tail_slots, 1)
            row["dequant_saved_ms"] = n_tail * out["dequant_ms_per_expert"]
            row["gemv_mtok1_ms"] = time_sync(lambda tk=tail_k: gemv_tail(tk, "swiglu", 1), samples=2)
            for mtok in (2, 4):
                row[f"gemv_mtok{mtok}_upper_ms"] = time_sync(
                    lambda tk=tail_k, m=mtok: gemv_tail(tk, "swiglu", m), samples=2)
        crossover_rows.append(row)
        print(f"  tail<={tail_k}: {row['tail_experts']} experts / {row['tail_slots']} slots | "
              f"dequant saved {row.get('dequant_saved_ms', 0):.2f} ms | gemv swiglu "
              f"{row.get('gemv_swiglu_ms', 0):.2f} ms (mtok2 upper "
              f"{row.get('gemv_mtok2_upper_ms', 0):.2f}, mtok4 "
              f"{row.get('gemv_mtok4_upper_ms', 0):.2f})", flush=True)
    out["crossover"] = crossover_rows
    entry["crossover"] = out


def phase_gemv_curve(hidden, topk_ids, store, plan, entry) -> None:
    """GEMV time vs S on this layer's real expert ids (the cost curve)."""
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        gemv_slots, materialized_ids, swiglu_gemv_slots)

    device = hidden.device
    selected = plan["selected_by_rank"]
    curve = []
    for s in (10, 40, 80, 160, 320, 512):
        if s > selected.numel():
            continue
        ids = materialized_ids(selected[:1].repeat(s), device) if s > selected.numel() \
            else materialized_ids(selected[:s], device)
        x = hidden[:1].repeat(s, 1)
        swiglu = time_sync(lambda: swiglu_gemv_slots(store, ids, x), samples=2)
        two = time_sync(lambda: F.silu(gemv_slots(store, "gate_proj", ids, x))
                        * gemv_slots(store, "up_proj", ids, x), samples=2)
        curve.append({"S": s, "swiglu_ms": swiglu, "two_gemv_ms": two,
                      "swiglu_us_per_slot": 1e3 * swiglu / s,
                      "two_gemv_us_per_slot": 1e3 * two / s})
        print(f"  gemv S={s:4d}: swiglu {swiglu:6.3f} ms ({1e3 * swiglu / s:5.2f} us/slot) | "
              f"2xgemv {two:6.3f} ms ({1e3 * two / s:5.2f} us/slot)", flush=True)
    entry["gemv_curve"] = curve


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=REPO / "tools/qwen4exp_flashnext/bench_h22_prefill_moe_eager.json")
    parser.add_argument("--layers", type=str, default="0,3")
    parser.add_argument("--tokens", type=str, default="512,2048")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--supers", type=str, default="1,2,4,8")
    parser.add_argument("--tails", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--only", default="hist,decompose,candidates,crossover,curve")
    args = parser.parse_args()

    if EXT is None or not KERNEL.is_file():
        raise SystemExit("build the TPC kernels + pytorch extension first")
    torch.ops.load_library(str(EXT))
    if not torch.hpu.is_available():
        raise SystemExit("no HPU")
    layers = [int(v) for v in args.layers.split(",") if v.strip()]
    tokens_list = [int(v) for v in args.tokens.split(",") if v.strip()]
    supers = [int(v) for v in args.supers.split(",") if v.strip()]
    tails = [int(v) for v in args.tails.split(",") if v.strip()]
    only = {v.strip() for v in args.only.split(",") if v.strip()}

    log: dict = {
        "unit": "H22 Phase 1 (prefill MoE: 3-launch projection-first dequant, "
                "single bucket, tail GEMV crossover)",
        "checkpoint": str(CHECKPOINT),
        "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "geometry": {"hidden": HIDDEN, "num_experts": EXPERTS, "top_k": TOP_K,
                     "moe_inter": MOE_INTER, "chunk": args.chunk,
                     "bf16_MiB_per_projection_per_expert": BF16_MIB_PER_PROJ},
        "layers": {},
    }
    for layer in layers:
        key = f"L{layer}"
        print(f"\n===== layer {layer}: loading routed planes =====", flush=True)
        started = time.time()
        store = load_routed_planes(layer)
        torch.hpu.synchronize()
        print(f"  loaded in {time.time() - started:.1f}s", flush=True)
        log["layers"][key] = {}
        for tokens in tokens_list:
            regime = f"T={tokens}"
            entry: dict = {}
            hidden, topk_ids, topk_weights = real_routing(layer, tokens, args.seed)
            plan = _plan_for(topk_ids, args.chunk, store)
            counts = plan["counts_sorted"]
            entry["routing"] = histogram(counts)
            hist = entry["routing"]
            print(f"\n== {key} {regime}: U={hist['unique']} cap={hist['count_max']} "
                  f"p50={hist['count_p50']} count==1 {hist['exact']['1']} "
                  f"count==2 {hist['exact']['2']} count<=4 {hist['cumulative']['4']} "
                  f"({hist['slots_cumulative']['4']} slots) ==", flush=True)
            if "hist" in only:
                print("  exact:", {k: v for k, v in hist["exact"].items() if v}, "| slots:",
                      {k: v for k, v in hist["slots_exact"].items() if v}, flush=True)
                print("  cumulative experts:", {k: v for k, v in hist["cumulative"].items() if v},
                      "| slots:", {k: v for k, v in hist["slots_cumulative"].items() if v},
                      f"| >8 {hist['gt_8']} ({hist['slots_gt_8']} slots)", flush=True)

            def guarded(name, fn, *a):
                if name not in only:
                    return
                try:
                    fn(*a)
                except Exception as error:  # noqa: BLE001 - keep the other phases
                    entry[f"{name}_error"] = f"{type(error).__name__}: {error}"
                    print(f"  !! {name} failed: {type(error).__name__}: {error}", flush=True)

            reference = _h9a(hidden, topk_ids, topk_weights, store)
            guarded("decompose", phase_decompose, hidden, topk_ids, topk_weights, store,
                    plan, args.chunk, entry, log)
            guarded("candidates", phase_candidates, hidden, topk_ids, topk_weights, store,
                    reference, entry, args.chunk, supers, tails, log)
            guarded("crossover", phase_crossover, hidden, topk_ids, topk_weights, store,
                    plan, entry, tails)
            guarded("curve", phase_gemv_curve, hidden, topk_ids, store, plan, entry)
            log["layers"][key][regime] = entry
            args.json.write_text(json.dumps(log, indent=2) + "\n")
            del hidden, topk_ids, topk_weights, reference, plan
            torch.hpu.empty_cache() if hasattr(torch.hpu, "empty_cache") else None
        del store
    args.json.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\nwrote {args.json}")
    return 0


def _h9a(hidden, topk_ids, topk_weights, store):
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped, compact_expert_ids)

    unique, local = compact_expert_ids(topk_ids, store.num_experts)
    gate_w = store.dequant("gate_proj", unique)
    up_w = store.dequant("up_proj", unique)
    down_w = store.dequant("down_proj", unique)
    return apply_routed_experts_grouped(hidden, local, topk_weights, gate_w, up_w, down_w)


if __name__ == "__main__":
    raise SystemExit(main())
