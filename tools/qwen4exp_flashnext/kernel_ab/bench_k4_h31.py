"""Base vs k4 decode MoE GEMVs in one process: bits, error vs fp64, us/call, for S in (1, 10) and T=1."""
import os, sys, time, zlib
from pathlib import Path
K = Path(__file__).resolve().parent
REPO = K.parents[2]
os.environ["GC_KERNEL_PATH"] = str(REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so") + ":/usr/lib/habanalabs/libtpc_kernels.so"
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
import numpy as np, torch
import habana_frameworks.torch.core  # noqa
torch.ops.load_library(str(next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))))
ops = torch.ops.custom_op
dev = torch.device("hpu")
E, H, I = 512, 2560, 640
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
def e4m3(c):
    c = c.astype(np.int64); sign = np.where(c & 0x80, -1., 1.); ex = (c >> 3) & 15; m = c & 7
    return sign * np.where(ex == 0, m / 8 * 2. ** -6, (1 + m / 8) * 2. ** (ex - 7))
def planes(name, rows, k):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return (rng.integers(0, 256, (E, rows, k // 2), dtype=np.uint8), rng.integers(40, 70, (E, rows, k // 16), dtype=np.uint8),
            (rng.random(E, dtype=np.float32) * 1e-2 + 1e-3).astype(np.float32))
def dense(p, e):
    packed, scales, s2 = p
    w = np.stack([E2M1[packed[e] & 15], E2M1[packed[e] >> 4]], -1).reshape(packed.shape[1], -1)
    return w * np.repeat(e4m3(scales[e]), 16, 1) * float(s2[e])
def k4(x):
    T, Kd = x.shape; full = Kd // 512 * 512
    head = x[:, :full].reshape(T, full // 512, 128, 4).transpose(-1, -2).reshape(T, full)
    return torch.cat([head, x[:, full:]], 1).contiguous()
gate, up, down = planes("gate", I, H), planes("up", I, H), planes("down", H, I)
D = lambda p: [torch.from_numpy(a).to(dev) for a in p]
gd, ud, dd = D(gate), D(up), D(down)
rng = np.random.default_rng(7)
def timeit(fn, reps=300):
    for _ in range(10): fn()
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.hpu.synchronize(); return (time.perf_counter() - t) / reps * 1e6
for S in (10, 1):
    ids = torch.from_numpy(rng.choice(E, S, replace=False).astype(np.int32)).to(dev)
    xb = torch.from_numpy((rng.standard_normal((1, H)) * .5).astype(np.float32)).bfloat16()
    xa = torch.from_numpy((rng.standard_normal((S, I)) * .5).astype(np.float32)).bfloat16()
    xb_d, xb4_d, xa_d, xa4_d = xb.to(dev), k4(xb).to(dev), xa.to(dev), k4(xa).to(dev)
    sw = lambda: ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2(ids, *gd, *ud, xb_d)
    sw4 = lambda: ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2(ids, *gd, *ud, xb4_d)
    dn = lambda: ops.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2(ids, dd[0], dd[1], xa_d, dd[2])
    dn4 = lambda: ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2(ids, dd[0], dd[1], xa4_d, dd[2])
    a, b, c, d = (f().cpu() for f in (sw, sw4, dn, dn4))
    idsn = ids.cpu().numpy(); xr = xb.float().numpy().astype(np.float64)[0]; xar = xa.float().numpy().astype(np.float64)
    ref_sw = np.stack([(lambda g, u: g / (1 + np.exp(-g)) * u)(dense(gate, e) @ xr, dense(up, e) @ xr) for e in idsn])
    ref_dn = np.stack([dense(down, e) @ xar[s] for s, e in enumerate(idsn)])
    rel = lambda got, ref: float(np.abs(got.float().numpy() - ref).max() / np.abs(ref).max())
    print(f"S={S:2d} swiglu base {timeit(sw):7.1f} us  k4 {timeit(sw4):7.1f} us | bits differ {int((a.view(torch.int16)!=b.view(torch.int16)).sum())}/{a.numel()} "
          f"| err base {rel(a, ref_sw):.2e} k4 {rel(b, ref_sw):.2e}")
    print(f"S={S:2d} down   base {timeit(dn):7.1f} us  k4 {timeit(dn4):7.1f} us | bits differ {int((c.view(torch.int16)!=d.view(torch.int16)).sum())}/{c.numel()} "
          f"| err base {rel(c, ref_dn):.2e} k4 {rel(d, ref_dn):.2e}")

# Device time inside one compiled graph: 24 calls per graph launch, host dispatch amortized.
N = 24
S = 10
id_list = [torch.from_numpy(rng.choice(E, S, replace=False).astype(np.int32)).to(dev) for _ in range(N)]
xb = torch.from_numpy((rng.standard_normal((1, H)) * .5).astype(np.float32)).bfloat16()
xa = torch.from_numpy((rng.standard_normal((S, I)) * .5).astype(np.float32)).bfloat16()
xb_d, xb4_d, xa_d, xa4_d = xb.to(dev), k4(xb).to(dev), xa.to(dev), k4(xa).to(dev)
def many(op, planes_args, x):
    def f(*ids):
        acc = None
        for i in ids:
            y = op(i, *planes_args, x) if op in (ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2, ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2) else op(i, planes_args[0], planes_args[1], x, planes_args[2])
            acc = y if acc is None else acc + y
        return acc
    return torch.compile(f, backend="hpu_backend", fullgraph=True, dynamic=False)
cases = (("swiglu base", ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2, (*gd, *ud), xb_d),
         ("swiglu k4  ", ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2, (*gd, *ud), xb4_d),
         ("down base  ", ops.custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2, dd, xa_d),
         ("down k4    ", ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2, dd, xa4_d))
for name, op, pa, x in cases:
    g = many(op, pa, x)
    us = timeit(lambda: g(*id_list), reps=50) / N
    print(f"graph S={S} {name} {us:7.1f} us/call (device, {N} calls per graph)")
