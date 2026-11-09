# SPDX-License-Identifier: MIT
"""CPU study: cheap BF16 exp variants for p = exp(s - m), end-to-end RMS vs FP32 softmax.

Per-half exact maxima as in exp_lut_half (halves at `first`). Variants:
  lut      : bf16(exp(bf16(s - m)))            (what exp_lut_half computes, up to LUT rounding)
  schr_*   : BF16 bits = sat_u16(A * (s - m) + B + c), A = 128/ln2, B = 127*128 (Schraudolph);
             trunc / rne, c = constant shift that centres the mantissa error
  quad     : Schraudolph on 2^(t) with a quadratic mantissa correction in FP32 before packing
"""
import struct
import sys
from pathlib import Path

import torch

A = 128.0 / 0.6931471805599453
B = 127.0 * 128.0


def load(name):
    data = bytearray(Path(name).read_bytes())
    _, keys, valid, _, _ = struct.unpack_from("<8s4I", data)
    off, t = 24, []
    for shape in ((128, 128), (keys, 128), (keys, 128)):
        n = shape[0] * shape[1]
        t.append(torch.frombuffer(data, dtype=torch.bfloat16, count=n, offset=off).clone().reshape(shape))
        off += n * 2
    ref = torch.frombuffer(data, dtype=torch.float32, count=16384, offset=off).clone().reshape(128, 128)
    return t[0], t[1], t[2], ref, keys, valid


def bits_to_bf16(bits):
    return bits.to(torch.int32).clamp(0, 0x7f7f).to(torch.int16).view(torch.bfloat16)


def schraudolph(x, mode, c):
    t = A * x.double() + B + c
    t = torch.floor(t) if mode == "trunc" else torch.round(t)
    return bits_to_bf16(t)


def quad(x):
    # 2^y with y = x*log2(e): integer part to exponent, mantissa from f + k*f*(1-f) (k=0.3433 minimax-ish)
    y = x.double() * 1.4426950408889634
    n = torch.floor(y)
    f = y - n
    mant = f + 0.3433 * f * (1 - f) * 0 + (f - f * f) * 0.3466  # 2^f - 1 approx: f + 0.3466 f(1-f)
    t = (n + 127) * 128 + torch.round(mant * 128)
    return bits_to_bf16(t)


def logq(xf32, mode):
    # T = x*128/ln2 + 127*128 in FP32 (x = s - m exact in FP32); exponent from T >> 7,
    # mantissa from a 128-entry table of 2^(j/128) (j = T & 127), packed to BF16 bits.
    t = xf32.double() * A + B
    t = torch.floor(t) if mode == "floor" else torch.round(t)
    t = t.clamp(0, 0x7f7f)
    j = (t % 128)
    mant = torch.round((torch.pow(2.0, j / 128) - 1) * 128).clamp(0, 127)
    return bits_to_bf16(t - j + mant)


variants = {"lut_max_stride2": ("stride", 2), "lut_max_stride4": ("stride", 4), "lut_max_stride8": ("stride", 8),
            "lut": None, "logq_rne_f32arg": ("logq", "rne"), "logq_floor_f32arg": ("logq", "floor"), "schr_trunc": ("trunc", 0.0), "schr_rne": ("rne", 0.0),
            "schr_rne_c": ("rne", -7.7), "quad": "quad"}
for name in sys.argv[1:]:
    q, k, v, ref, keys, valid = load(name)
    k, v = k[:valid].float(), v[:valid].float()
    s = (q.float() @ k.T).bfloat16().float()
    first = (keys // 128 + 1) // 2 * 128
    halves = [(0, min(first, valid)), (min(first, valid), valid)]
    out = {}
    gaps = []
    for vname, spec in variants.items():
        num = torch.zeros(128, 128, dtype=torch.float64)
        den = torch.zeros(128, 1, dtype=torch.float64)
        mrow = s.max(-1, keepdim=True).values.double()
        for lo, hi in halves:
            sh = s[:, lo:hi]
            m = sh.max(-1, keepdim=True).values
            x = (sh - m).bfloat16()                       # BF16 subtract as on the TPC
            if isinstance(spec, tuple) and spec[0] == "stride":
                # Max over every k-th 128-column chunk only (m_hat <= m); exp of BF16 (s - m_hat).
                cols = torch.arange(sh.shape[1])
                pick = (cols // 128) % spec[1] == 0
                m = sh[:, pick].max(-1, keepdim=True).values
                x = (sh - m).bfloat16()
                gap = (sh.max(-1, keepdim=True).values - m)
                gaps.append(gap.max().item())
                p = torch.exp(x.float()).bfloat16()
            elif spec is None:
                p = torch.exp(x.float()).bfloat16()
            elif isinstance(spec, tuple) and spec[0] == "logq":
                p = logq(sh - m, spec[1])                   # FP32 subtract (no BF16 rounding of x)
            elif spec == "quad":
                p = quad(x.float())
            else:
                p = schraudolph(x.float(), *spec)
            pf = p.double()
            w = torch.exp(m.double() - mrow)
            o = (pf.float() @ v[lo:hi]).bfloat16().double()   # MME: FP32 acc, BF16 out
            num += o * w
            den += pf.sum(-1, keepdim=True) * w
        o = (num / den).float().bfloat16().float()
        out[vname] = ((o - ref).square().mean() / ref.square().mean()).sqrt().item()
        # per-element error of the exp itself on this data
    print(Path(name).stem, " ".join(f"{k}={v * 100:.4f}%" for k, v in out.items()),
          "max_gap(m - m_hat) per stride:", [round(g, 2) for g in gaps], flush=True)
