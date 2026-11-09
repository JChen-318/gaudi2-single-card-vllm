# SPDX-License-Identifier: MIT
# CPU-only: SageAttention2-style FP8 P.V precision on real H3 QKV, per 128-row tile.
# S = FP16(q.k) (MME FP16 scores); per key quarter m = exact row max of S;
# P~ = exp(S - m) in (0, 1] (FP32), then variants:
#   bf16      : P BF16, V BF16, FP32 accumulate, row sum of BF16 P   (current-path proxy)
#   p8_vbf16  : P e4m3(P~*448)/448, V BF16                           (P quantisation only)
#   sage2     : P e4m3(P~*448)/448, V e4m3 per-channel scale (max|V_c|/448 per head)
#   sage2_sumq: as sage2, row sum from quantised P (numerator/denominator consistent)
# Quarters merged exactly with exp(m_q - M). Error: RMS vs FP32 softmax per 128-row tile.
import argparse, json, math, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+", type=Path)
ap.add_argument("--tiles", type=int, default=4, help="random 128-row tiles per head")
ap.add_argument("--heads", default="all")
ap.add_argument("--output", type=Path)
a = ap.parse_args()
torch.set_num_threads(32)
torch.manual_seed(0)
E4 = torch.float8_e4m3fn


def q8(x):
    return x.clamp(-448, 448).to(E4).float()


def run_tile(qs, k, v, vsc, v8):
    s = (qs.float() @ k.float().T).half().float()           # [128, L] FP16 scores
    L = k.shape[0]
    first = (((L + 127) // 128 * 128) + 511) // 512 * 128
    bounds = [(0, first), (first, 2 * first), (2 * first, 3 * first), (3 * first, L)]
    ref = torch.softmax(qs.float() @ k.float().T, -1) @ v.float()   # FP32 reference
    outs = {}
    for name in ("bf16", "p8_vbf16", "sage2", "sage2_sumq"):
        parts = []
        for lo, hi in bounds:
            sh = s[:, lo:hi]
            m = sh.max(-1, keepdim=True).values
            pt = torch.exp(sh - m)                              # (0, 1]
            if name == "bf16":
                p = pt.bfloat16().float(); vv = v[lo:hi].float(); o = (p @ vv); l = p.sum(-1, keepdim=True)
            else:
                p8 = q8(pt * 448.0) / 448.0
                if name == "p8_vbf16":
                    o = p8 @ v[lo:hi].float()
                else:
                    o = (p8 @ v8[lo:hi]) * vsc
                l = (p8 if name == "sage2_sumq" else pt).sum(-1, keepdim=True)
            parts.append((m, o.bfloat16().float(), l))
        M = torch.stack([x[0] for x in parts]).amax(0)
        num = sum(o * torch.exp(mm - M) for mm, o, _ in parts)
        den = sum(l * torch.exp(mm - M) for mm, _, l in parts)
        out = (num / den).bfloat16().float()
        outs[name] = float(((out - ref).square().sum() / ref.square().sum()).sqrt())
    return outs


summary = {}
for dump in a.dumps:
    t0 = time.time()
    d = torch.load(dump, map_location="cpu", weights_only=True, mmap=True)
    scale = float(d["scale"])
    Q, K, V = d["q"][0], d["k"][0], d["v"][0]
    L, H, D = Q.shape
    heads = range(H) if a.heads == "all" else [int(x) for x in a.heads.split(",")]
    rows = []
    for h in heads:
        q, k, v = Q[:, h], K[:, h], V[:, h]
        vsc = v.float().abs().amax(0).clamp_min(1e-30) / 448.0     # per-channel [D]
        v8 = q8(v.float() / vsc)
        starts = torch.randint(0, L // 128, (a.tiles,)).tolist()
        for st in starts:
            lo = st * 128
            r = run_tile((q[lo:lo + 128].float() * scale).bfloat16(), k, v, vsc, v8)
            rows.append(dict(head=h, tile=st, **r))
        print(json.dumps(dict(dump=dump.name, head=h, **{k_: max(x[k_] for x in rows if x["head"] == h)
                               for k_ in ("bf16", "p8_vbf16", "sage2", "sage2_sumq")})), flush=True)
    s = {}
    for k_ in ("bf16", "p8_vbf16", "sage2", "sage2_sumq"):
        vals = [x[k_] for x in rows]
        s[k_] = dict(max=max(vals), mean=sum(vals) / len(vals), over_1pct=sum(x >= .01 for x in vals),
                     tiles=len(vals), worst_head=max(rows, key=lambda x: x[k_])["head"])
    summary[dump.name] = s
    print("FP8PV_LAYER " + json.dumps({dump.name: s, "seconds": round(time.time() - t0, 1)}), flush=True)
print("FP8PV_SUMMARY " + json.dumps(summary), flush=True)
if a.output:
    a.output.write_text(json.dumps(summary, indent=1) + "\n")
