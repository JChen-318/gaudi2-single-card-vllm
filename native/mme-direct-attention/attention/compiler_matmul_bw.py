# SPDX-License-Identifier: MIT
"""Stock Synapse compiler reference for the QK output-side question (2026-11-09).

C[M, N] = A[M, K] @ B[K, N] through plain torch.matmul on HPU (eager, compiler-chosen
geometry and slicing, output in HBM). K is swept at constant output; two M values give a
size slope that cancels per-op launch overhead. A HBM fill of the same bytes is a write
ceiling control. Prints one JSON line per case.
"""
import argparse, json, time
import torch
import habana_frameworks.torch.core  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=9472)
p.add_argument("--ms", default="8192,32768")
p.add_argument("--ks", default="32,64,128,256,512")
p.add_argument("--iters", type=int, default=30)
p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
p.add_argument("--pv", action="store_true", help="PV geometry bmm [b,512,Lk]@[b,Lk,128] instead of the K sweep")
p.add_argument("--pv-keys", default="9472,37760")
p.add_argument("--pv-transposed", action="store_true", help="with --pv: O^T = V^T @ P^T (128x512 output)")
a = p.parse_args()
dt = getattr(torch, a.dtype)
ms = [int(x) for x in a.ms.split(",")]
ks = [int(x) for x in a.ks.split(",")]
torch.manual_seed(0)


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.hpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.hpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


# Exactness spot check on a small shape (small integers, exact in bf16/fp16 accumulation).
x = torch.randint(-2, 3, (256, 128)).to(dt); y = torch.randint(-2, 3, (128, 1024)).to(dt)
exact = bool(torch.equal((x.to("hpu") @ y.to("hpu")).cpu().float(), x.float() @ y.float()))
print("CMM_EXACT " + json.dumps({"exact": exact}), flush=True)

for m in ms:
    out = torch.empty((m, a.n), dtype=dt, device="hpu")
    us = timed(lambda: out.fill_(1.0), a.iters)
    print("CMM " + json.dumps({"case": "fill", "m": m, "n": a.n, "out_bytes": out.numel() * 2,
                               "us": us, "TBps_write": out.numel() * 2 / us / 1e6}), flush=True)
    del out

if a.pv:
    # PV geometry: O[b] = P[b][512, Lk] @ V[b][Lk, 128]; batch slope cancels launch overhead.
    for lk in [int(x) for x in a.pv_keys.split(",")]:
        r = {}
        for bt in (8, 32):
            if a.pv_transposed:  # O^T[128, 512] = V^T[128, Lk] @ P^T[Lk, 512]: same FLOPs, 512-wide output
                P = torch.randn(bt, 128, lk, dtype=dt, device="hpu"); V = torch.randn(bt, lk, 512, dtype=dt, device="hpu")
                O = torch.empty((bt, 128, 512), dtype=dt, device="hpu")
            else:
                P = torch.randn(bt, 512, lk, dtype=dt, device="hpu"); V = torch.randn(bt, lk, 128, dtype=dt, device="hpu")
                O = torch.empty((bt, 512, 128), dtype=dt, device="hpu")
            us = timed(lambda: torch.bmm(P, V, out=O), a.iters)
            fl = 2 * bt * 512 * lk * 128
            r[bt] = (us, fl)
            print("CMM " + json.dumps({"case": "pv_bmm", "batch": bt, "keys": lk, "us": us, "TFLOPS": fl / us / 1e6,
                                       "us_per_512q": us / bt}), flush=True)
            del P, V, O
        (u0, f0), (u1, f1) = r[8], r[32]
        print("CMM_SLOPE " + json.dumps({"case": "pv_bmm", "keys": lk, "slope_TFLOPS": (f1 - f0) / (u1 - u0) / 1e6,
                                         "slope_us_per_512q": (u1 - u0) / 24}), flush=True)
    print("CMM_DONE", flush=True)
    raise SystemExit(0)

res = {}
for k in ks:
    for m in ms:
        A = torch.randn(m, k, dtype=dt, device="hpu"); B = torch.randn(k, a.n, dtype=dt, device="hpu")
        C = torch.empty((m, a.n), dtype=dt, device="hpu")
        us = timed(lambda: torch.matmul(A, B, out=C), a.iters)
        ob = m * a.n * 2
        res[(k, m)] = (us, ob)
        print("CMM " + json.dumps({"case": "matmul", "k": k, "m": m, "n": a.n, "out_bytes": ob, "us": us,
                                   "TBps_out": ob / us / 1e6, "TFLOPS": 2 * m * a.n * k / us / 1e6}), flush=True)
        del A, B, C
    if len(ms) >= 2:
        (u0, b0), (u1, b1) = res[(k, ms[0])], res[(k, ms[-1])]
        print("CMM_SLOPE " + json.dumps({"k": k, "slope_TBps_out": (b1 - b0) / (u1 - u0) / 1e6,
                                         "slope_TFLOPS": 2 * (ms[-1] - ms[0]) * a.n * k / (u1 - u0) / 1e6,
                                         "intercept_us": u0 - b0 / ((b1 - b0) / (u1 - u0))}), flush=True)
print("CMM_DONE", flush=True)
