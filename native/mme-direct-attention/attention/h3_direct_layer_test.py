# SPDX-License-Identifier: MIT
"""One full H3 attention layer (all 56 heads) through h3_direct_attention vs the
production max_free_attention, on a real QKV dump. Isolated process, card idle.

Reports per-layer times (steady state, excludes the first call that prepares the
plan / compiles graphs), per-head gate results, RMS of direct vs production for
every head, and RMS vs a CPU FP32 softmax reference on sampled query rows."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
import torch
import habana_frameworks.torch.core  # noqa: F401
import h3_direct_attention as hd

p = argparse.ArgumentParser()
p.add_argument("dump", type=Path)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--ref-heads", default=os.environ.get("H3_LAYER_REF_HEADS","0,27,55"))
p.add_argument("--ref-rows", type=int, default=512)
p.add_argument("--full-ref-heads", default=os.environ.get("H3_LAYER_FULL_REF_HEADS",""))
a = p.parse_args()
torch.set_num_threads(16)
try:
    import vllm_gaudi.omni.attention as attn
    production = attn.max_free_attention
    prod_name = "vllm_gaudi.omni.attention.max_free_attention"
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"cannot import production attention: {e!r}")

d = torch.load(a.dump, map_location="cpu", weights_only=True, mmap=True)
scale = float(d["scale"])
q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to("hpu") for n in ("q", "k", "v"))   # [1, H, L, D]
torch.hpu.synchronize()
# Keep the full allocations alive while diagnosing one head, to preserve the
# same HBM addressing conditions as the failed full-layer run.
head_limit=int(os.environ.get("H3_LAYER_HEAD_LIMIT","0"))
if head_limit:
    q,k,v=(x[:,:head_limit] for x in (q,k,v))




def timed(fn, n, fingerprint=False):
    out, times, phases, hashes = None, [], [], []
    for _ in range(n):
        before=dict(hd.STATS)
        torch.hpu.synchronize(); t = time.perf_counter()
        out = fn(); torch.hpu.synchronize()
        times.append((time.perf_counter() - t) * 1000)
        phases.append({k:hd.STATS[k]-before[k] for k in before})
        if fingerprint:
            hashes.append(hashlib.sha256(out.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest())
    return out, times, phases, hashes


direct_fn = lambda: hd.direct_max_free_attention(q, k, v, scale, production)  # noqa: E731
prod_fn = lambda: production(q, k, v, scale)  # noqa: E731
out_d, t_d, phases_d, hashes_d = timed(direct_fn, a.repeats + 1, fingerprint=True)
out_p, t_p, _, _ = timed(prod_fn, a.repeats + 1)
gates = hd.head_gates(q[0] * (hd.INPUT_Q_SCALE if hd.POW2_Q else scale), k[0], v[0], hd.SCORE_FACTOR)
dd, pp = out_d[0].float(), out_p[0].float()
rms_dp = ((dd - pp).square().mean((1, 2)) / pp.square().mean((1, 2)).clamp_min(1e-30)).sqrt().cpu().tolist()
ref = {}
qc, kc, vc = (d[n][0].float() for n in ("q", "k", "v"))       # [L, H, D]
rows = torch.linspace(0, q.shape[2] - 1, a.ref_rows).long()
od, op = out_d[0].cpu().float(), out_p[0].cpu().float()
ref_heads=range(q.shape[1]) if a.ref_heads=="all" else [int(x) for x in a.ref_heads.split(",") if int(x)<q.shape[1]]
for h in ref_heads:
    s = (qc[rows, h] @ kc[:, h].T) * scale
    r = torch.softmax(s, -1) @ vc[:, h]
    e = lambda o: float(((o[h, rows] - r).square().mean() / r.square().mean()).sqrt())  # noqa: E731
    ref[h] = dict(direct=e(od), production=e(op))
full_ref={}
for h in [int(x) for x in a.full_ref_heads.split(",") if x]:
    assert 0<=h<q.shape[1]
    error_sum=reference_sum=0.;tiles=[]
    for lo in range(0,q.shape[2],128):
        r=torch.softmax((qc[lo:lo+128,h]@kc[:,h].T)*scale,-1)@vc[:,h]
        es=float((od[h,lo:lo+128]-r).square().sum());rs=float(r.square().sum())
        error_sum+=es;reference_sum+=rs;tiles.append((es/max(rs,1e-30))**.5)
    full_ref[h]=dict(rows=q.shape[2],rms=(error_sum/reference_sum)**.5,max_tile_rms=max(tiles),
                     tiles_over_one_percent=sum(x>=.01 for x in tiles),tile_rms=tiles)
print("H3_LAYER " + json.dumps(dict(
    dump=str(a.dump), pow2_q=hd.POW2_Q, heads=q.shape[1], rows=q.shape[2], production=prod_name,
    direct_ms=t_d, production_ms=t_p,
    direct_steady_ms=statistics.median(t_d[1:]), production_steady_ms=statistics.median(t_p[1:]),
    direct_phase_deltas=phases_d, direct_output_sha256=hashes_d,
    gates_failed=[h for h, g in enumerate(gates) if not g], stats=hd.STATS,
    rms_direct_vs_production_max=max(rms_dp), rms_direct_vs_production=rms_dp,
    rms_vs_fp32_reference=ref, rms_vs_fp32_full_heads=full_ref, finite=bool(torch.isfinite(out_d).all()))), flush=True)

assert bool(torch.isfinite(out_d).all()), "Direct layer output is not finite"
assert len(set(hashes_d))==1, "Repeated direct outputs differ"
assert ref and max(r["direct"] for r in ref.values())<.01, "Direct output exceeds FP32 reference RMS limit"

assert all(r["max_tile_rms"]<.01 for r in full_ref.values()), "Full-reference 128-query tile exceeds RMS limit"
