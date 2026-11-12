# SPDX-License-Identifier: MIT
"""GitHub issue #3 (gaudi2-single-card-vllm): numerical fallback and variable valid length.

Isolated process, card idle. Needs H3_ATTENTION_OVERLAY = the fixed vllm_gaudi/omni/attention.py.
 1. direct path, real layer at L=37750: output SHA must equal the pre-fix value (--expect-sha).
 2. variable valid length: L=37760 (rows appended), 37700 and 28800 (truncated) through the direct path,
    then back to 37750: every head direct, error vs exact attention and vs a CPU FP32 reference on sampled
    rows, and the 37750 SHA must come back unchanged (set_valid rewrite + stale-tail clearing).
 3. issue counterexamples inside a real 56-head call: head 0 anti-parallel, head 1 orthogonal large-norm Q/K,
    V = 1. The direct gates must reject both, the output must be 1, the other heads bit-identical to test 1.
 4. max_free_attention with the guard vs the deployed (pre-fix) function: counterexamples (old -> 0, new -> 1)
    and the real layer (must be bit-identical), with timing.
 5. automatic key-capacity buckets (H3DIRECT_KEYS=auto).
 6. issue #4: injected device-write failures during a valid-length change (restore, and poisoned plan).
"""
import argparse
import hashlib
import json
import math
import os
import runpy
import time
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
import vllm_gaudi.omni.attention as attn

p = argparse.ArgumentParser()
p.add_argument("dump", type=Path)
p.add_argument("--expect-sha", default="")
p.add_argument("--ref-heads", default="0,27,55")
p.add_argument("--ref-rows", type=int, default=256)
a = p.parse_args()
torch.set_num_threads(16)

old = runpy.run_path(attn.__file__, run_name="pre_fix_attention")       # pristine deployed definitions
overlay = os.environ["H3_ATTENTION_OVERLAY"]
exec(compile(open(overlay, encoding="utf-8").read(), overlay, "exec"), attn.__dict__)
assert hasattr(attn, "exact_attention") and attn.HPU_SDPA_MAX_FREE_GUARD
import h3_direct_attention as hd  # noqa: E402

results = {}
sha = lambda t: hashlib.sha256(t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()  # noqa: E731
d = torch.load(a.dump, map_location="cpu", weights_only=True, mmap=True)
scale = float(d["scale"])
Q, K, V = (d[n].permute(0, 2, 1, 3).contiguous() for n in ("q", "k", "v"))   # CPU [1, H, L, D]
H, L = Q.shape[1], Q.shape[2]
ref_heads = [int(x) for x in a.ref_heads.split(",")]


def direct(q, k, v):
    torch.hpu.synchronize()
    before = dict(hd.STATS)
    out = hd.direct_max_free_attention(q, k, v, scale, attn.max_free_attention, attn.exact_attention)
    torch.hpu.synchronize()
    return out, {key: hd.STATS[key] - before[key] for key in ("direct_heads", "fallback_heads", "calls")}


def rel(x, y):
    x, y = x.float(), y.float()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


def fp32_ref_error(q, k, v, out):
    rows = torch.linspace(0, q.shape[2] - 1, a.ref_rows).long()
    worst = 0.0
    for h in ref_heads:
        qs = q[0, h, rows].float() * scale
        ref = torch.softmax(qs @ k[0, h].float().T, -1) @ v[0, h].float()
        worst = max(worst, rel(out[0, h, rows].cpu(), ref))
    return worst


# 1. Baseline length through the direct path.
q, k, v = (x.to("hpu") for x in (Q, K, V))
out, st = direct(q, k, v)
out, st = direct(q, k, v)                        # steady state (first call prepares)
base_sha, base_out = sha(out), out.cpu()
results["t1_baseline"] = dict(L=L, sha=base_sha, expect=a.expect_sha or None,
                              sha_ok=(base_sha == a.expect_sha) if a.expect_sha else None, **st)
print("ISSUE3 t1 " + json.dumps(results["t1_baseline"]), flush=True)

# 2. Variable valid lengths, then back.
t2_sha = {}
for n in [int(x) for x in os.environ.get("ISSUE3_LENGTHS", "37760,37700,28800").split(",")] + [L]:
    if n >= L:
        extra = n - L
        q2, k2, v2 = (torch.cat([x, x[:, :, :extra]], dim=2) for x in (Q, K, V))
    else:
        q2, k2, v2 = (x[:, :, :n].contiguous() for x in (Q, K, V))
    qh, kh, vh = (x.to("hpu") for x in (q2, k2, v2))
    t = time.perf_counter()
    o, st = direct(qh, kh, vh)
    wall = time.perf_counter() - t
    row = dict(L=n, wall_s=round(wall, 3), **st)
    t2_sha[n] = sha(o)
    if n == L:
        row["sha"] = sha(o); row["sha_back_to_baseline"] = row["sha"] == base_sha
    else:
        ex = attn.exact_attention(qh, kh, vh, scale)
        row["rel_vs_exact"] = rel(o, ex)
        row["fp32_ref_worst_rel"] = fp32_ref_error(q2, k2, v2, o)
    results[f"t2_L{n}"] = row
    print("ISSUE3 t2 " + json.dumps(row), flush=True)
    del qh, kh, vh

# 3. Issue counterexamples inside a real 56-head call.
torch.manual_seed(0)
u = torch.nn.functional.normalize(torch.randn(128), dim=0)
qc, kc, vc = Q.clone(), K.clone(), V.clone()
big = 40.0
qc[0, 0] = (big * u).to(qc.dtype); kc[0, 0] = (-big * u).to(kc.dtype)          # anti-parallel
e1, e2 = torch.zeros(128), torch.zeros(128); e1[0] = 1; e2[1] = 1
qc[0, 1] = (big * e1).to(qc.dtype); kc[0, 1] = (big * e2).to(kc.dtype)          # orthogonal
vc[0, :2] = 1
qh, kh, vh = (x.to("hpu") for x in (qc, kc, vc))
o, st = direct(qh, kh, vh)
oc = o.cpu()
results["t3_counterexamples_direct"] = dict(
    **st, head0_max_err=float((oc[0, 0].float() - 1).abs().max()), head1_max_err=float((oc[0, 1].float() - 1).abs().max()),
    other_heads_bit_identical=bool(torch.equal(oc[0, 2:], base_out[0, 2:])))
print("ISSUE3 t3 " + json.dumps(results["t3_counterexamples_direct"]), flush=True)

# 6. Issue #4: a failed valid-length update must leave a consistent plan. Test-only native fault injection
#    (context_quad_test_fail_write) fails the Nth device write of the next set_valid.
#    a. 2nd of 4 writes fails (slot 0 exp already rewritten): native restores; the plan forgets its length; L
#       again needs no rebuild and must give the t1 SHA (it would differ if slot 0 kept the new exp code).
#    b. same failure with the restore failing too: native poisons the plan (-ENOTRECOVERABLE) and refuses
#       submits; the next call rebuilds it and must give the t1 SHA.
#    c. the injected-to length afterwards: same SHA as in test 2.
T6_LEN = int(os.environ.get("ISSUE4_LENGTH", "37700"))
if T6_LEN in t2_sha and hd._PLAN is not None and hd._PLAN.valid == L:
    import ctypes as C
    inject = hd._PLAN.lib.context_quad_test_fail_write
    inject.argtypes, inject.restype = [C.c_uint, C.c_int], C.c_int
    submit_probe = hd._PLAN.lib.context_quad_submit2
    submit_probe.argtypes, submit_probe.restype = [C.c_uint, C.c_int], C.c_int
    q6 = tuple(x[:, :, :T6_LEN].contiguous().to("hpu") for x in (Q, K, V))
    for case, restore_fails in (("a_restored", 0), ("b_poisoned", 1)):
        plan = hd._PLAN
        assert plan.valid == L
        inject(2, restore_fails)
        ok = plan.set_valid(T6_LEN)
        row = dict(set_valid_ok=ok, plan_valid_after=plan.valid)
        row["native_set_valid_L_rc"] = plan._set_valid(L)       # a: 0 (already L, no writes); b: -ENOTRECOVERABLE
        if row["native_set_valid_L_rc"] == -131:                  # only probe submit when it cannot queue anything
            row["submit_rc_while_poisoned"] = submit_probe(0, 0)
        sw = hd.STATS["plan_switches"]
        o, st = direct(q, k, v)
        row.update(st, rebuilds=hd.STATS["plan_switches"] - sw, sha_equals_t1=sha(o) == base_sha)
        results[f"t6_{case}"] = row
        print("ISSUE4 t6 " + json.dumps({case: row}), flush=True)
    o, st = direct(*q6)
    results["t6_c_new_length"] = dict(L=T6_LEN, **st, sha_equals_t2=sha(o) == t2_sha[T6_LEN])
    print("ISSUE4 t6 " + json.dumps(results["t6_c_new_length"]), flush=True)
    o, st = direct(q, k, v)
    results["t6_d_back"] = dict(**st, sha_equals_t1=sha(o) == base_sha)
    print("ISSUE4 t6 " + json.dumps(results["t6_d_back"]), flush=True)
    del q6, o

# 5. Automatic key-capacity buckets (H3DIRECT_KEYS=auto): cross every bucket in both directions. A length must
#    give the same result whatever ran before (37750 -> the t1 SHA again after a rebuild).
if hd.KEYS_SETTING == "auto":
    for n in [int(x) for x in os.environ.get("ISSUE3_SWITCH", "37859,38317,39000,39500,40000,40500,41000,41344,41472,37750").split(",")]:
        if n >= L:
            q2, k2, v2 = (torch.cat([x, x[:, :, :n - L]], dim=2) for x in (Q, K, V))
        else:
            q2, k2, v2 = (x[:, :, :n].contiguous() for x in (Q, K, V))
        qh, kh, vh = (x.to("hpu") for x in (q2, k2, v2))
        before = (hd.STATS["plan_switches"], hd.STATS["plan_prepare_s"])
        t = time.perf_counter()
        o, st = direct(qh, kh, vh)
        row = dict(L=n, bucket=hd.select_bucket(n), plan_keys=hd._PLAN.keys if hd._PLAN else None, wall_s=round(time.perf_counter() - t, 3),
                   switches=hd.STATS["plan_switches"] - before[0],
                   prepare_s=round(hd.STATS["plan_prepare_s"] - before[1], 2), **st)
        if n == L:
            row["sha"] = sha(o); row["sha_equals_t1"] = row["sha"] == base_sha
        else:
            row["fp32_ref_worst_rel"] = fp32_ref_error(q2, k2, v2, o)
        results.setdefault("t5_buckets", []).append(row)
        print("ISSUE3 t5 " + json.dumps(row), flush=True)
        del qh, kh, vh, o

# 4. max_free_attention guard vs the deployed function.
two = (qc[:, :2].to("hpu"), kc[:, :2].to("hpu"), vc[:, :2].to("hpu"))
o_old = old["max_free_attention"](*two, scale).cpu().float()
o_new = attn.max_free_attention(*two, scale).cpu().float()
results["t4_counterexamples_maxfree"] = dict(old_max_err=float((o_old - 1).abs().max()), new_max_err=float((o_new - 1).abs().max()),
                                             old_finite=bool(torch.isfinite(o_old).all()))
print("ISSUE3 t4a " + json.dumps(results["t4_counterexamples_maxfree"]), flush=True)


def timed(fn, n=3):
    times, outv = [], None
    for _ in range(n):
        torch.hpu.synchronize(); t = time.perf_counter(); outv = fn(); torch.hpu.synchronize()
        times.append(round((time.perf_counter() - t) * 1000, 2))
    return outv, times


o_old, t_old = timed(lambda: old["max_free_attention"](q, k, v, scale))
o_new, t_new = timed(lambda: attn.max_free_attention(q, k, v, scale))
results["t4_real_layer_maxfree"] = dict(bit_identical=bool(torch.equal(o_old.cpu(), o_new.cpu())), old_ms=t_old, new_ms=t_new)
print("ISSUE3 t4b " + json.dumps(results["t4_real_layer_maxfree"]), flush=True)
print("ISSUE3_RESULTS " + json.dumps(results), flush=True)
