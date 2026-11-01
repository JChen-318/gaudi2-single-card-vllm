"""Large-matrix MME calibration: python3 gemm_cal.py"""
import subprocess, threading, time
import torch
import habana_frameworks.torch as htorch  # noqa: F401
import habana_frameworks.torch.core as htcore  # noqa: F401

DEV = "hpu"
BF = torch.bfloat16
FP8 = torch.float8_e4m3fn


def timed(name, macs, fn, peak=216e12, n=12):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.hpu.synchronize()
        ts.append(time.perf_counter() - t)
    best, mid = min(ts[1:]), sorted(ts[1:])[len(ts) // 2]
    print(f"{name:34} best {best * 1e3:8.2f} ms median {mid * 1e3:8.2f} ms  {macs / best / 1e12:7.1f} T MAC/s  {100 * macs / best / peak:5.1f}% of peak", flush=True)


def rnd(*shape, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(BF).to(DEV)


def to8(x):
    return torch.ops.hpu.cast_to_fp8_v2(x, 1.0, False, False, FP8)[0]


def fp8gemm(a, b):
    return torch.ops.hpu.fp8_gemm_v2(A=a, trans_A=False, B=b, trans_B=False, D=None, out_dtype=BF, A_scale_inv=1.0,
                                     B_scale_inv=1.0, bias=None, accumulate=False)


for m in (8192, 16384, 24576):
    a, b = rnd(m, m), rnd(m, m, seed=2)
    torch.hpu.synchronize()
    timed(f"bf16 {m}^3", m ** 3, lambda: torch.matmul(a, b))
    fa, fb = to8(a), to8(b)
    torch.hpu.synchronize()
    timed(f"fp8  {m}^3", m ** 3, lambda: fp8gemm(fa, fb), peak=432e12)
    del a, b, fa, fb
# only the shape of the attention: wide output, narrow inner dimension
for rows, keys, inner in ((12288, 147456, 256), (12288, 147456, 1024), (12288, 36864, 1024)):
    a, b = rnd(rows, inner), rnd(inner, keys, seed=2)
    torch.hpu.synchronize()
    timed(f"bf16 {rows}x{inner}x{keys}", rows * inner * keys, lambda: torch.matmul(a, b))
    del a, b
