# SPDX-License-Identifier: MIT
# CPU-only: compare the deployed FP32-affine logq exp (exp_f16_logq_ms8) with a
# proposed FP16-affine variant, over EVERY block of a batch/full-head input.
#   current: t = rhne(f32(min(s,ceil)) * 184.66 + (16256 - m*184.66)); clamp >=128
#   f16:     d = f16(min(s,ceil) - m); q = rhne(f16(d * f16(184.66))); q = max(q, -16128)
#            bits = q + T[q & 127], T[j] = 16256 + mant(j) - j  (bias folded into LUT)
# Both: FP16 scores, sampled (1/8) half max, BF16 P, FP32 row sums.
import argparse, json, struct
from pathlib import Path
import torch

torch.set_num_threads(16)
ap = argparse.ArgumentParser()
ap.add_argument("input", type=Path)
ap.add_argument("--output", type=Path)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=1, help="check every Nth block")
args = ap.parse_args()
b = bytearray(args.input.read_bytes())
_, keys, valid, dim, n, rows, reserved = struct.unpack_from("<8s6I", b)
off, ts = 32, []
for shape in [(n * 128, 128), (keys, 128), (keys, 128)]:
    c = shape[0] * shape[1]
    ts.append(torch.frombuffer(b, dtype=torch.bfloat16, count=c, offset=off).clone().reshape(shape)); off += 2 * c
q, k, v = ts
ref = torch.frombuffer(b, dtype=torch.float32, count=n * 16384, offset=off).clone().reshape(n, 128, 128)
valid_q = reserved or rows
first = (keys // 128 + 1) // 2 * 128
C32 = 184.6649652337873
C16 = torch.tensor(C32, dtype=torch.float16)
j = torch.arange(128)
mant = torch.round((2 ** (j.float() / 128) - 1) * 128).int()
T = 16256 + mant - j            # folded table (int16 range)
kf, vf = k.float(), v.float()

def probs(sh, m16, mode):
    """sh: FP16 scores [r, w]; m16: sampled max (FP16) [r, 1]."""
    m = m16.float()
    ceil = (m + 64).half()
    x = torch.minimum(sh, ceil)
    if mode == "current":
        t = torch.round(x.float() * C32 + (16256 - m * C32))     # FP32 affine, rhne
        t = t.clamp(min=128).int()
        jj = t & 127
        bits = t - jj + mant[jj]
    else:
        d = (x.float() - m16.float()).half()                      # f16 sub (one rounding)
        p = (d.float() * C16.float()).half()                      # f16 mul (one rounding)
        qq = torch.round(p.float()).clamp(min=-16128).int()       # cvt f16->i16 rhne, i16 max
        bits = qq + T[qq & 127]
    return bits.to(torch.int16).view(torch.bfloat16)

out_rows = []
blocks = range(0, n if not args.limit else min(n, args.limit), args.stride)
worst = {"current": 0.0, "f16": 0.0}
for i in blocks:
    if i * 128 >= valid_q:
        continue
    qi = q[i * 128:(i + 1) * 128].float()
    raw = qi @ kf.T
    s16 = raw.half()
    rec = {"block": i}
    for mode in ("current", "f16"):
        num = torch.zeros(128, 128); den = torch.zeros(128, 1)
        parts = []
        for lo, hi in [(0, first), (first, valid)]:
            sh = s16[:, lo:hi]
            picks = (torch.arange(sh.shape[1]) // 128) % 8 == 0
            m16 = sh[:, picks].max(-1, keepdim=True).values
            pb = probs(sh, m16, mode).float()
            parts.append((m16.float(), (pb @ vf[lo:hi]).bfloat16().float(), pb.sum(-1, keepdim=True)))
        M = torch.maximum(parts[0][0], parts[1][0])
        for m, o, l in parts:
            w = torch.exp(m - M); num += o * w; den += l * w
        out = (num / den).bfloat16().float()
        r = ref[i]
        rows_valid = min(128, valid_q - i * 128)
        e = float(((out[:rows_valid] - r[:rows_valid]).square().mean() / r[:rows_valid].square().mean()).sqrt())
        rec[mode] = e; worst[mode] = max(worst[mode], e)
    out_rows.append(rec)
    if len(out_rows) % 20 == 0:
        print(json.dumps(dict(done=len(out_rows), worst=worst)), flush=True)
summary = dict(input=str(args.input), blocks=len(out_rows), worst=worst,
               mean={m: sum(r[m] for r in out_rows) / len(out_rows) for m in worst})
print("SIM_SUMMARY " + json.dumps(summary), flush=True)
if args.output:
    args.output.write_text(json.dumps(dict(summary=summary, rows=out_rows), indent=1) + "\n")
