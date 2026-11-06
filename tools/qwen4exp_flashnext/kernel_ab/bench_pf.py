"""bench_pf.py LIB OUT.npz: decode MoE k4 GEMVs from one kernel library: device us/call in a graph, output bits.

Run once per library (GC_KERNEL_PATH is read at start-up) and compare the npz files for bit equality.
"""
import os, sys, time, zlib
from pathlib import Path
os.environ["GC_KERNEL_PATH"] = str(Path(sys.argv[1]).resolve()) + ":/usr/lib/habanalabs/libtpc_kernels.so"
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
REPO = Path(__file__).resolve().parents[3]
import numpy as np, torch
import habana_frameworks.torch.core  # noqa
torch.ops.load_library(str(next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))))
ops = torch.ops.custom_op
dev = torch.device("hpu")
E, H, I = 512, 2560, 640
def planes(name, rows, k):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return [torch.from_numpy(a).to(dev) for a in (
        rng.integers(0, 256, (E, rows, k // 2), dtype=np.uint8), rng.integers(40, 70, (E, rows, k // 16), dtype=np.uint8),
        (rng.random(E, dtype=np.float32) * 1e-2 + 1e-3).astype(np.float32))]
gd, ud, dd = planes("gate", I, H), planes("up", I, H), planes("down", H, I)
rng = np.random.default_rng(7)
def k4(x):
    T, Kd = x.shape; full = Kd // 512 * 512
    head = x[:, :full].reshape(T, full // 512, 128, 4).transpose(-1, -2).reshape(T, full)
    return torch.cat([head, x[:, full:]], 1).contiguous()
def timeit(fn, reps=50):
    for _ in range(5): fn()
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.hpu.synchronize(); return (time.perf_counter() - t) / reps * 1e6
SW = ops.custom_qwen4exp_nvfp4_indexed_swiglu_gemv_k4_bf16_gaudi2
DN = ops.custom_qwen4exp_nvfp4_indexed_gemv_k4_bf16_gaudi2
out = {}
N = 24
for T, S in ((1, 10), (2, 20), (1, 1)):
    ids = [torch.from_numpy(rng.choice(E, S, replace=False).astype(np.int32)).to(dev) for _ in range(N)]
    xb = k4(torch.from_numpy((rng.standard_normal((T, H)) * .5).astype(np.float32)).bfloat16()).to(dev)
    xa = k4(torch.from_numpy((rng.standard_normal((S, I)) * .5).astype(np.float32)).bfloat16()).to(dev)
    def sw_many(*ii):
        return [SW(i, *gd, *ud, xb) for i in ii]
    def dn_many(*ii):
        return [DN(i, dd[0], dd[1], xa, dd[2]) for i in ii]
    gs = torch.compile(sw_many, backend="hpu_backend", fullgraph=True, dynamic=False)
    gn = torch.compile(dn_many, backend="hpu_backend", fullgraph=True, dynamic=False)
    out[f"sw_T{T}_S{S}"] = torch.stack(gs(*ids)).cpu().view(torch.int16).numpy()
    out[f"dn_T{T}_S{S}"] = torch.stack(gn(*ids)).cpu().view(torch.int16).numpy()
    out[f"sw_eager_T{T}_S{S}"] = SW(ids[0], *gd, *ud, xb).cpu().view(torch.int16).numpy()
    out[f"dn_eager_T{T}_S{S}"] = DN(ids[0], dd[0], dd[1], xa, dd[2]).cpu().view(torch.int16).numpy()
    ts = timeit(lambda: gs(*ids)) / N
    tn = timeit(lambda: gn(*ids)) / N
    print(f"T={T} S={S:2d}: swiglu {ts:6.1f} us/call  down {tn:6.1f} us/call (graph, {N} calls)", flush=True)
np.savez(sys.argv[2], **out)
