# SPDX-License-Identifier: Apache-2.0
"""H23 Phase 1 micro-probes: the four candidate primitives, on Gaudi2.

No checkpoint, no layer: pure op-level questions the H23 report needs answered
before any production change.

1. ``torch.ops.hpu.rms_norm`` exists here? Its numerics vs ``gemma_rmsnorm``
   (which is what ``qsa_math.gemma_rmsnorm`` runs) and its latency at the three
   QSA shapes: indexer q ``[T, 4, 128]``, pooled blocks ``[B, 128]``.
2. ``torch.argsort(descending=True, stable=True)`` == ``argsort(-x, stable=True)``
   on HPU? (``_packed_block_topk``'s tie rule; one launch saved per layer.)
3. Merged ``[12288+640, 2560]`` GEMM vs two GEMMs: bit-identical? faster?
4. The selection path at S_kv = 2560 (640 blocks, k = 512): where the 1.4 ms goes.

Run:
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/probe_h23_ops.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
sys.path.insert(0, str(REPO / "vllm-gaudi"))

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_qwen4exp_gc_kernel_path",
    REPO / "vllm-gaudi/vllm_gaudi/models/qwen4exp/_gc_kernel_path.py")
_gc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gc)
_gc.ensure_gc_kernel_path()

import torch  # noqa: E402
import habana_frameworks.torch  # noqa: E402,F401

DEV = "hpu"


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def bench(fn, warmup=5, samples=20, iters=50):
    """Pipelined per-call latency in ms.

    A synchronize around *one* call measures the launch+sync round trip
    (~150-250 us on this card, see bench_layer_e2e's attribution note), not
    the op. Repeating the call ``iters`` times before the single synchronize
    keeps the queue full and divides the round trip out, which is how a
    decode step actually issues these ops.
    """
    for _ in range(warmup):
        fn()
    torch.hpu.synchronize()
    ts = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.hpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3 / iters)
    return _median(ts)


def probe_rmsnorm(results):
    from vllm_gaudi.models.qwen4exp.qsa_math import gemma_rmsnorm

    have = hasattr(torch.ops.hpu, "rms_norm")
    print(f"[1] torch.ops.hpu.rms_norm present: {have}", flush=True)
    results["hpu_rms_norm_available"] = bool(have)
    if not have:
        return
    print("    doc:", (torch.ops.hpu.rms_norm.__doc__ or "").strip().splitlines()[:1], flush=True)
    torch.manual_seed(0)
    for name, shape in (("indexer_q_T1", (1, 4, 128)), ("indexer_q_T512", (512, 4, 128)),
                        ("pooled_B640", (640, 128))):
        x = (torch.randn(*shape) * 0.5).bfloat16().to(DEV)
        w = torch.randn(128).to(DEV)
        eps = 1e-6
        ref = gemma_rmsnorm(x, w, eps)
        got = torch.ops.hpu.rms_norm(x, w, eps, None, True)[0]
        got_plain = torch.ops.hpu.rms_norm(x, (1.0 + w.float()).to(torch.bfloat16), eps)[0]
        diff = float((ref.float() - got.float()).abs().max())
        exact = torch.equal(ref, got)
        t_ref = bench(lambda: gemma_rmsnorm(x, w, eps))
        t_hpu = bench(lambda: torch.ops.hpu.rms_norm(x, w, eps, None, True)[0])
        diff_plain = float((ref.float() - got_plain.float()).abs().max())
        print(f"    {name:14s} absmax(addOne)={diff:.3e} bit_equal={exact} "
              f"absmax(prefold)={diff_plain:.3e} "
              f"eager={t_ref*1e3:8.2f} us  hpu_rms_norm={t_hpu*1e3:8.2f} us", flush=True)
        results[f"rmsnorm_{name}"] = {"absmax": diff, "bit_equal": bool(exact),
                                      "eager_us": t_ref * 1e3, "hpu_us": t_hpu * 1e3}


def probe_argsort(results):
    print("[2] argsort descending-stable vs negated stable", flush=True)
    torch.manual_seed(3)
    ok = True
    max_width = 0
    for trial in range(30):
        width = 640 if trial % 2 else 2048
        max_width = max(max_width, width)
        # Realistic score distribution: relu of a dot product, many exact ties.
        scores = torch.relu(torch.randn(1, width, device=DEV)).to(torch.float32)
        scores = (scores * 100).round() / 100  # force exact ties
        a = torch.argsort(-scores, dim=-1, stable=True)
        b = torch.argsort(scores, dim=-1, stable=True, descending=True)
        same = torch.equal(a, b)
        ok = ok and same
        if not same:
            print(f"    MISMATCH at trial {trial} width {width}", flush=True)
            break
    x = torch.relu(torch.randn(1, 640, device=DEV)).round()
    t_a = bench(lambda: torch.argsort(-x, dim=-1, stable=True))
    t_b = bench(lambda: torch.argsort(x, dim=-1, stable=True, descending=True))
    print(f"    identical over 30 trials: {ok}; negate+sort={t_a*1e3:.2f} us "
          f"sort(desc)={t_b*1e3:.2f} us", flush=True)
    results["argsort_desc_stable_equal"] = bool(ok)
    results["argsort_neg_us"] = t_a * 1e3
    results["argsort_desc_us"] = t_b * 1e3


def probe_merge(results):
    import torch.nn.functional as F

    print("[3] merged qkv+index GEMM", flush=True)
    torch.manual_seed(0)
    hidden = 2560
    qkv_n, idx_n = 12288, 640
    for tokens in (1, 4, 512):
        x = (torch.randn(tokens, hidden) * 0.1).bfloat16().to(DEV)
        wq = (torch.randn(qkv_n, hidden) * 0.02).bfloat16().to(DEV)
        wi = (torch.randn(idx_n, hidden) * 0.02).bfloat16().to(DEV)
        wm = torch.cat((wq, wi), dim=0).contiguous()
        a = F.linear(x, wq)
        b = F.linear(x, wi)
        m = F.linear(x, wm)
        eq = torch.equal(m[:, :qkv_n], a) and torch.equal(m[:, qkv_n:], b)
        t_two = bench(lambda: (F.linear(x, wq), F.linear(x, wi)))
        t_one = bench(lambda: F.linear(x, wm))
        print(f"    T={tokens:4d} bit_equal={eq} two={t_two*1e3:8.2f} us "
              f"merged={t_one*1e3:8.2f} us  ({t_two - t_one:+.2f} us)", flush=True)
        results[f"merge_T{tokens}"] = {"bit_equal": bool(eq), "two_us": t_two * 1e3,
                                       "merged_us": t_one * 1e3}


def probe_selection(results):
    from vllm_gaudi.models.qwen4exp import qsa_math

    print("[4] selection at S_kv=2560 (640 blocks, k=512)", flush=True)
    torch.manual_seed(0)
    ratio, budget, heads, dim = 4, 2048, 4, 128
    tokens, total_blocks = 1, 640
    q = (torch.randn(tokens, heads, dim) * 0.5).bfloat16().to(DEV)
    pooled = (torch.randn(total_blocks, dim) * 0.5).bfloat16().to(DEV)
    positions = torch.tensor([total_blocks * ratio - 1], device=DEV, dtype=torch.long)

    def full():
        return qsa_math.select_qsa_indices_from_pooled(
            q, pooled, positions, total_blocks=total_blocks,
            token_budget=budget, compress_ratio=ratio, total_tokens=total_blocks * ratio)

    idx = full()
    t_full = bench(full)
    print(f"    select_qsa_indices_from_pooled  T=1 unpadded : {t_full*1e3:8.2f} us", flush=True)

    window = torch.cat((pooled, pooled.new_zeros(2048 - total_blocks, dim)))
    positions_w = positions

    def win():
        return qsa_math.select_qsa_indices_from_pooled(
            q, window, positions_w, total_blocks=total_blocks,
            token_budget=budget, compress_ratio=ratio, total_tokens=total_blocks * ratio)

    idx_w = win()
    t_win = bench(win)
    print(f"    select_..._from_pooled         T=1 2048 pad : {t_win*1e3:8.2f} us "
          f"(same indices: {torch.equal(idx, idx_w)})", flush=True)

    out = torch.full((tokens, budget + ratio - 1), -1, dtype=torch.int32, device=DEV)
    n_complete = torch.div(positions + 1, ratio, rounding_mode="floor")
    qf = q.float()
    pooled_t = pooled.float().transpose(0, 1).contiguous()
    scores = torch.relu(torch.matmul(qf, pooled_t)).sum(dim=1) / (dim ** 0.5)

    def score_only():
        return torch.relu(torch.matmul(qf, pooled.float().transpose(0, 1))).sum(dim=1) / (dim ** 0.5)

    t_score = bench(score_only)
    print(f"    score plane (float+transpose+matmul+relu+sum): {t_score*1e3:8.2f} us", flush=True)

    valid = torch.arange(total_blocks, device=DEV).unsqueeze(0) < n_complete.unsqueeze(1)
    t_mask = bench(lambda: scores.masked_fill(~valid, float("-inf")))
    t_sort = bench(lambda: torch.argsort(-scores, dim=-1, stable=True))
    order = torch.argsort(-scores, dim=-1, stable=True)
    k = 512
    t_pack = bench(lambda: qsa_math._packed_block_topk(q, pooled, n_complete, total_blocks, k, ratio, out))
    print(f"    masked_fill={t_mask*1e3:7.2f} us  argsort={t_sort*1e3:7.2f} us  "
          f"_packed_block_topk(total)={t_pack*1e3:7.2f} us", flush=True)

    # Candidate: descending stable sort (no negation).
    def packed_desc():
        dots = torch.matmul(q.float(), pooled.float().transpose(0, 1))
        s = torch.relu(dots).sum(dim=1) / (dim ** 0.5)
        block_ids = torch.arange(pooled.shape[0], device=q.device)
        vb = block_ids.unsqueeze(0) < n_complete.unsqueeze(1)
        s = s.masked_fill(~vb, float("-inf"))
        order = torch.argsort(s, dim=-1, stable=True, descending=True)
        picked = order[:, :k]
        ok = vb.gather(1, picked)
        offs = torch.arange(ratio, device=q.device)
        bt = picked.unsqueeze(-1) * ratio + offs
        bt = bt.masked_fill(~ok.unsqueeze(-1), -1)
        flat = bt.reshape(tokens, k * ratio)
        o = torch.full((tokens, budget + ratio - 1), -1, dtype=torch.int32, device=q.device)
        o[:, : k * ratio] = flat.to(torch.int32)
        ntok = ok.sum(dim=-1) * ratio
        tail_off = torch.arange(ratio - 1, device=q.device)
        tail_pos = n_complete.unsqueeze(-1) * ratio + tail_off
        tail_ok = tail_off.unsqueeze(0) < (positions + 1 - n_complete * ratio).unsqueeze(1)
        tail_cols = ntok.unsqueeze(-1) + tail_off
        src = torch.where(tail_ok, tail_pos, torch.full_like(tail_pos, -1))
        o.scatter_(1, tail_cols, src.to(torch.int32))
        return o

    cand = packed_desc()
    t_cand = bench(packed_desc)
    print(f"    candidate (desc-sort)          : {t_cand*1e3:8.2f} us  identical={torch.equal(idx, cand)}",
          flush=True)
    results["select_unpadded_us"] = t_full * 1e3
    results["select_window2048_us"] = t_win * 1e3
    results["select_score_us"] = t_score * 1e3
    results["select_pack_us"] = t_pack * 1e3
    results["select_candidate_us"] = t_cand * 1e3
    results["select_candidate_identical"] = bool(torch.equal(idx, cand))


def probe_attention(results):
    from vllm_gaudi.models.qwen4exp import qsa_math

    print("[5] attention paths (exact fp32 planes vs BF16 MME fast)", flush=True)
    torch.manual_seed(0)
    heads, kv_heads, dim = 24, 2, 256
    scale = dim ** -0.5
    cases = [
        ("decode_T1_S257", 1, 257, "dense"),
        ("decode_T4_S257", 4, 257, "dense"),
        ("decode_T1_S2560_sparse", 1, 2560, "sparse"),
        ("prefill_T512_S512", 512, 512, "dense"),
    ]
    for name, tokens, s_kv, kind in cases:
        q = (torch.randn(tokens, heads, dim) * 0.5).bfloat16().to(DEV)
        k = (torch.randn(s_kv, kv_heads, dim) * 0.5).bfloat16().to(DEV)
        v = (torch.randn(s_kv, kv_heads, dim) * 0.5).bfloat16().to(DEV)
        if kind == "dense":
            fn = lambda: qsa_math.dense_prefix_attention(q, k, v, scale)  # noqa: E731
        else:
            width = 2048 + 3
            cols = torch.arange(width, device=DEV)
            idx = torch.where(cols.unsqueeze(0) < s_kv, cols.unsqueeze(0),
                              torch.full_like(cols.unsqueeze(0), -1)).to(torch.int32)
            fn = lambda: qsa_math.sparse_attention(q, k, v, idx, scale)  # noqa: E731
        import os as _os
        _os.environ["VLLM_QWEN4EXP_QSA_EXACT"] = "1"
        exact_out = fn()
        t_exact = bench(fn)
        _os.environ["VLLM_QWEN4EXP_QSA_EXACT"] = "0"
        fast_out = fn()
        t_fast = bench(fn)
        _os.environ.pop("VLLM_QWEN4EXP_QSA_EXACT")
        diff = float((exact_out.float() - fast_out.float()).abs().max())
        rel = diff / max(1e-9, float(exact_out.float().abs().max()))
        print(f"    {name:22s} exact={t_exact*1e3:8.1f} us  fast={t_fast*1e3:8.1f} us  "
              f"({t_exact/max(t_fast,1e-9):4.2f}x)  absmax={diff:.3e} rel={rel:.2e}", flush=True)
        results[f"attn_{name}"] = {"exact_us": t_exact * 1e3, "fast_us": t_fast * 1e3,
                                   "absmax": diff, "rel": rel}


def probe_sync_cost(results):
    print("[6] host sync / bound-check cost", flush=True)
    x = torch.arange(1, device=DEV, dtype=torch.long)
    big = torch.randn(2051, device=DEV)
    t_item = bench(lambda: int(x.max()), iters=20)
    t_item2 = bench(lambda: int(x[0].item()), iters=20)
    t_tolist = bench(lambda: x.tolist(), iters=20)
    t_big = bench(lambda: int(big.max()), iters=20)
    # A dependent chain: a GEMM whose result must be read back.
    a = torch.randn(1, 2560, device=DEV).bfloat16()
    w = torch.randn(12288, 2560, device=DEV).bfloat16()
    import torch.nn.functional as F
    t_gemm = bench(lambda: F.linear(a, w), iters=50)
    t_gemm_read = bench(lambda: int(F.linear(a, w).float().sum()), iters=20)
    print(f"    int(x.max())={t_item*1e3:7.1f} us  int(x[0])={t_item2*1e3:7.1f} us  "
          f"tolist(1)={t_tolist*1e3:7.1f} us  int(big.max())={t_big*1e3:7.1f} us", flush=True)
    print(f"    GEMM only={t_gemm*1e3:7.1f} us   GEMM + host read={t_gemm_read*1e3:7.1f} us "
          f"(read costs {t_gemm_read*1e3 - t_gemm*1e3:.1f} us)", flush=True)
    results["sync"] = {"max_item_us": t_item * 1e3, "index_item_us": t_item2 * 1e3,
                       "tolist_us": t_tolist * 1e3, "big_max_us": t_big * 1e3,
                       "gemm_us": t_gemm * 1e3, "gemm_read_us": t_gemm_read * 1e3}


def bench_pair(fa, fb, warmup=3, pairs=11, iters=150):
    """Interleaved paired A/B (alternating order) in ms per call.

    This card's clock drifts a few percent between blocks, so two sequential
    ``bench`` calls can invert a 10% difference. Alternating the order inside
    one measurement window cancels the drift and reports both medians plus
    the median of the per-pair ratio.
    """
    def one(fn):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.hpu.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters

    for _ in range(warmup):
        fa()
        fb()
    torch.hpu.synchronize()
    ta, tb, ratios = [], [], []
    for index in range(pairs):
        if index % 2 == 0:
            a, b = one(fa), one(fb)
        else:
            b, a = one(fb), one(fa)
        ta.append(a)
        tb.append(b)
        ratios.append(a / max(b, 1e-9))
    return _median(ta), _median(tb), _median(ratios)


def probe_h28(results):
    """[7] H28: untiled sparse gather and one MRoPE composition per step."""
    from vllm_gaudi.models.qwen4exp import qsa_math
    from vllm_gaudi.models.qwen4exp.control import (
        bind_host_control,
        cached_mrope_cos_sin,
        invalidate_mrope_cos_sin,
        reset_host_control,
    )

    print("[7] H28: sparse gather tiling + MRoPE composition", flush=True)
    torch.manual_seed(0)
    heads, kv_heads, dim = 24, 2, 256
    scale = dim ** -0.5
    tokens, s_kv, width = 1, 2560, 2048 + 3
    q = (torch.randn(tokens, heads, dim) * 0.5).bfloat16().to(DEV)
    k = (torch.randn(s_kv, kv_heads, dim) * 0.5).bfloat16().to(DEV)
    v = (torch.randn(s_kv, kv_heads, dim) * 0.5).bfloat16().to(DEV)
    cols = torch.arange(width, device=DEV)
    idx = torch.where(cols.unsqueeze(0) < s_kv, cols.unsqueeze(0),
                      torch.full_like(cols.unsqueeze(0), -1)).to(torch.int32)

    old_budget = qsa_math._FAST_TILE_BYTES
    per_col = tokens * kv_heads * dim * 2
    # Reproduce the pre-H28 spelling: a 1024-column cap on top of the 1 GiB
    # budget, i.e. 1024 + 1024 + 3 key tiles at this shape. The tile is read
    # from the module global at call time, so each variant sets it before its
    # own call (a module-global store costs nothing next to the GEMM).
    capped_tile = max(1, (1024 * per_col) // per_col)

    def exact_fn():
        return qsa_math._sparse_attention_queries(q, k, v, idx, scale)

    def fast_fn():
        qsa_math._FAST_TILE_BYTES = old_budget
        return qsa_math._sparse_attention_fast(q, k, v, idx, scale)

    def tiled_fn():
        qsa_math._FAST_TILE_BYTES = 1024 * per_col
        return qsa_math._sparse_attention_fast(q, k, v, idx, scale)

    exact_out = exact_fn()
    fast_out = fast_fn()
    tiled_out = tiled_fn()
    qsa_math._FAST_TILE_BYTES = old_budget
    # Paired, alternating-order A/B: exact vs H28 untiled vs the old tiling.
    t_exact, t_fast, r_ef = bench_pair(exact_fn, fast_fn)
    t_fast2, t_tiled, r_ft = bench_pair(fast_fn, tiled_fn)
    t_exact2, t_tiled2, r_et = bench_pair(exact_fn, tiled_fn)
    # The variants set the module global before each call, so the last timed
    # call leaves the capped budget behind: restore it before reading it.
    qsa_math._FAST_TILE_BYTES = old_budget
    diff = float((exact_out.float() - fast_out.float()).abs().max())
    rel = diff / max(1e-9, float(exact_out.float().abs().max()))
    tiled_diff = float((tiled_out.float() - fast_out.float()).abs().max())
    tile_now = qsa_math._fast_width_tile(tokens, kv_heads, dim)
    print(f"    sparse T=1 S_kv={s_kv} width={width} "
          f"(tile now {tile_now} cols > width -> one gather):", flush=True)
    print(f"      exact fp32 plane      {t_exact*1e3:8.1f} / {t_exact2*1e3:8.1f} us",
          flush=True)
    print(f"      fast untiled (H28)    {t_fast*1e3:8.1f} / {t_fast2*1e3:8.1f} us  "
          f"paired exact/fast {r_ef:4.2f}x  absmax={diff:.3e} rel={rel:.2e}", flush=True)
    print(f"      fast tiled @{capped_tile:4d} cols   {t_tiled*1e3:8.1f} / "
          f"{t_tiled2*1e3:8.1f} us  "
          f"paired exact/tiled {r_et:4.2f}x  paired tiled/fast {r_ft:4.2f}x  "
          f"absmax vs untiled={tiled_diff:.3e}", flush=True)
    results["h28_sparse_decode"] = {
        "exact_us": t_exact * 1e3, "exact_us_second": t_exact2 * 1e3,
        "fast_untiled_us": t_fast * 1e3, "fast_untiled_us_second": t_fast2 * 1e3,
        "fast_tiled_us": t_tiled * 1e3, "fast_tiled_us_second": t_tiled2 * 1e3,
        "paired_ratio_exact_over_fast": r_ef,
        "paired_ratio_exact_over_tiled": r_et,
        "paired_ratio_tiled_over_fast": r_ft,
        "tile_cols_now": tile_now, "tiled_cols": capped_tile,
        "absmax_exact_vs_fast": diff, "rel": rel,
        "absmax_untiled_vs_tiled": tiled_diff,
    }

    # MRoPE composition: 12 layers of one step compose the same rows.
    rotary_dim = 64
    cache = _cos_sin_table(262144, rotary_dim, base=1e7).bfloat16().to(DEV)
    section = [11, 11, 10]
    rows = torch.arange(1, device=DEV).unsqueeze(0).expand(3, -1).contiguous()
    long_rows = torch.arange(512, device=DEV).unsqueeze(0).expand(3, -1).contiguous()
    for name, pos in (("T1", rows), ("T512", long_rows)):
        general = lambda: qsa_math.mrope_cos_sin(  # noqa: E731
            cache, pos, mrope_section=section, mrope_interleaved=True,
            equal_rows=False)
        shortcut = lambda: qsa_math.mrope_cos_sin(  # noqa: E731
            cache, pos, mrope_section=section, mrope_interleaved=True,
            equal_rows=True)
        same = torch.equal(general(), shortcut())
        t_general = bench(general)
        t_short = bench(shortcut)
        # Twelve layers of one bound step: one composition, eleven reuses.
        token = bind_host_control(pos, None, None)
        try:
            invalidate_mrope_cos_sin()
            key = (id(cache), cache.dtype, cache.device, tuple(cache.shape),
                   tuple(section), True, tuple(pos.shape))
            calls = {"n": 0}

            def counting():
                calls["n"] += 1
                return general()

            first = cached_mrope_cos_sin(pos, key, counting)
            for _ in range(11):
                cached_mrope_cos_sin(pos, key, counting)
            composed = calls["n"]
            equal_cached = torch.equal(first, general())

            def twelve():
                for _ in range(12):
                    cached_mrope_cos_sin(pos, key, general)

            def twelve_plain():
                for _ in range(12):
                    general()

            t_step = bench(twelve)
            t_step_plain = bench(twelve_plain)
        finally:
            reset_host_control(token)
            invalidate_mrope_cos_sin()
        print(f"    mrope {name:5s} general={t_general*1e3:7.2f} us "
              f"equal-rows={t_short*1e3:7.2f} us (bit_equal={same}) | "
              f"12 layers plain={t_step_plain*1e3:7.2f} us "
              f"cached={t_step*1e3:7.2f} us "
              f"(compositions/12 layers={composed}, bit_equal={equal_cached})", flush=True)
        results[f"h28_mrope_{name}"] = {
            "general_us": t_general * 1e3, "equal_rows_us": t_short * 1e3,
            "equal_rows_bit_equal": bool(same),
            "step_12_plain_us": t_step_plain * 1e3,
            "step_12_cached_us": t_step * 1e3,
            "step_12_compositions": composed,
            "step_12_cached_bit_equal": bool(equal_cached),
        }


def _cos_sin_table(max_pos: int, rotary_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def main() -> int:
    results: dict = {"probe": "probe_h23_ops", "lazy": os.environ.get("PT_HPU_LAZY_MODE")}
    probe_rmsnorm(results)
    probe_argsort(results)
    probe_merge(results)
    probe_selection(results)
    probe_attention(results)
    probe_sync_cost(results)
    probe_h28(results)
    out = Path(__file__).resolve().parent / "probe_h23_ops.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
