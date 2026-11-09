# SPDX-License-Identifier: MIT
"""CPU simulation: end-to-end error of exp variants on a real G2ATTN1 block.

current : p = bf16(exp(fp32(S) - b))          (device path today)
bf16arg : p = bf16(exp(bf16(S - b)))          (what a BF16-input exp such as exp_bf16 would see)
Also reports how loose the Cauchy-Schwarz bound is (b - rowmax).
"""
import struct
import sys
from pathlib import Path

import torch

for name in sys.argv[1:]:
    data = bytearray(Path(name).read_bytes())
    magic, keys, valid, dim, rows = struct.unpack_from("<8s4I", data)
    off = 24
    t = []
    for shape in ((128, 128), (keys, 128), (keys, 128)):
        n = shape[0] * shape[1]
        t.append(torch.frombuffer(data, dtype=torch.bfloat16, count=n, offset=off).clone().reshape(shape))
        off += n * 2
    q, k, v = t
    ref = torch.frombuffer(data, dtype=torch.float32, count=16384, offset=off).clone().reshape(128, 128)
    k, v = k[:valid], v[:valid]
    s = (q.float() @ k.float().T).bfloat16().float()          # MME output S is BF16
    b = (q.float().norm(dim=-1) * k.float().norm(dim=-1).max()).bfloat16().float()[:, None]
    gap = (b - s.max(dim=-1, keepdim=True).values).squeeze()

    def finish(p):
        p = p.bfloat16()
        l = p.float().sum(-1, keepdim=True)
        o = (p.float() @ v.float()).bfloat16().float() / l
        o = o.bfloat16().float()
        return ((o - ref).square().mean() / ref.square().mean()).sqrt().item()

    cur = finish(torch.exp(s - b))
    bfa = finish(torch.exp((s - b).bfloat16().float()))
    m = s.max(dim=-1, keepdim=True).values                     # exact row max of BF16 S
    tma = finish(torch.exp((s - m).bfloat16().float()))        # two-pass: max, then BF16-arg exp
    print(f"{Path(name).stem}: bound gap b-rowmax min/median/max = "
          f"{gap.min():.2f}/{gap.median():.2f}/{gap.max():.2f}; "
          f"rms current={cur:.6f} bf16-arg={bfa:.6f} rowmax+bf16-arg={tma:.6f}")
