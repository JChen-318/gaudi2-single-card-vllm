# SPDX-License-Identifier: MIT
# CPU-only: H3 outlier-layer analysis for an exact-row-max direct path.
# Per head: fixed0 gate inputs (score bound, |v| max), real score range on
# sampled rows, and the error of the modelled kernel
#   S = FP16(q.k)  ->  per key quarter: m = exact row max (FP16 values)
#   P = BF16 exp via the FP16 affine logq method (exp_f16_logq.c F16_AFFINE,
#       ceiling m+64 never active with an exact max), FP32 row sums,
#   quarters merged with exp(m_q - M) (normalize_quad)
# against FP32 softmax. Rows: --rows uniform rows per head, plus every row of
# --full-heads in 128-row tiles (the 1% gate is per 128-row tile).
import argparse, json, math
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dump", type=Path)
ap.add_argument("--rows", type=int, default=256)
ap.add_argument("--full-heads", default="")
ap.add_argument("--output", type=Path)
a = ap.parse_args()
torch.set_num_threads(32)
d = torch.load(a.dump, map_location="cpu", weights_only=True, mmap=True)
scale = float(d["scale"])
Q, K, V = d["q"][0], d["k"][0], d["v"][0]          # [L, H, D] bf16
L, H, D = Q.shape
KEYS = (L + 127) // 128 * 128
FIRST = (KEYS + 511) // 512 * 128
bounds = [(0, FIRST), (FIRST, 2 * FIRST), (2 * FIRST, 3 * FIRST), (3 * FIRST, L)]
C16 = torch.tensor(184.6649652337873, dtype=torch.float16).float()
j = torch.arange(128)
T = 16256 + torch.round((2 ** (j.float() / 128) - 1) * 128).int() - j


def model(qs, kf, vf):
    """qs [r, D] bf16 scaled; kf/vf [L, D] float. Returns [r, D]."""
    s16 = (qs.float() @ kf.T).half()
    if not torch.isfinite(s16).all():
        return None
    parts = []
    for lo, hi in bounds:
        sh = s16[:, lo:hi]
        m = sh.max(-1, keepdim=True).values                   # exact FP16 max
        dd = (sh.float() - m.float()).half()
        p = (dd.float() * C16).half()
        qq = torch.round(p.float()).clamp(min=-16128).int()
        bits = (qq + T[qq & 127]).to(torch.int16).view(torch.bfloat16).float()
        parts.append((m.float(), (bits @ vf[lo:hi]).bfloat16().float(), bits.sum(-1, keepdim=True)))
    M = torch.stack([x[0] for x in parts]).amax(0)
    num = sum(o * torch.exp(m - M) for m, o, _ in parts)
    den = sum(l * torch.exp(m - M) for m, _, l in parts)
    return (num / den).bfloat16().float()


def rms(o, r):
    return float(((o - r).square().sum() / r.square().sum()).sqrt())


heads = []
rows = torch.linspace(0, L - 1, a.rows).long()
for h in range(H):
    q, k, v = Q[:, h], K[:, h], V[:, h]
    qs = q * scale
    kf, vf = k.float(), v.float()
    bound = float(qs.float().norm(dim=-1).max() * kf.norm(dim=-1).max())
    vmax = float(vf.abs().max())
    s = qs[rows].float() @ kf.T
    smax, smin = float(s.max()), float(s.min())
    ref = torch.softmax(s, -1) @ vf
    out = model(qs[rows], kf, vf)
    rec = dict(head=h, bound=bound, vmax=vmax, score_max=smax, score_min=smin,
               fp16_ok=out is not None, rms=None if out is None else rms(out, ref))
    heads.append(rec)
    print(json.dumps(rec), flush=True)
full = {}
for h in [int(x) for x in a.full_heads.split(",") if x]:
    q, k, v = Q[:, h], K[:, h], V[:, h]
    qs, kf, vf = q * scale, k.float(), v.float()
    tiles = []
    for lo in range(0, L, 128):
        r = torch.softmax(qs[lo:lo + 128].float() @ kf.T, -1) @ vf
        o = model(qs[lo:lo + 128], kf, vf)
        tiles.append(float("inf") if o is None else rms(o, r))
    full[h] = dict(max_tile=max(tiles), tiles_over_1pct=sum(t >= .01 for t in tiles), tiles=len(tiles))
    print(json.dumps({"full_head": h, **full[h]}), flush=True)
summary = dict(dump=str(a.dump), heads=H, rows_per_head=a.rows,
               fixed0_gate_fail=[r["head"] for r in heads if r["bound"] > 80],
               max_bound=max(r["bound"] for r in heads), max_score=max(r["score_max"] for r in heads),
               fp16_overflow_heads=[r["head"] for r in heads if not r["fp16_ok"]],
               max_rms=max((r["rms"] for r in heads if r["rms"] is not None), default=None),
               worst_heads=sorted((r for r in heads if r["rms"] is not None), key=lambda r: -r["rms"])[:5],
               full=full)
print("OUTLIER_SUMMARY " + json.dumps(summary), flush=True)
if a.output:
    a.output.write_text(json.dumps(dict(summary=summary, heads=heads), indent=1) + "\n")
