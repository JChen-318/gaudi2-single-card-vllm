import os, sys, time, zlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.environ["GC_KERNEL_PATH"] = str(REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so") + ":/usr/lib/habanalabs/libtpc_kernels.so"
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
sys.path.insert(0, str(REPO / "vllm-gaudi"))
import numpy as np, torch
import habana_frameworks.torch.core  # noqa
torch.ops.load_library(str(next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))))
from vllm_gaudi.models.qwen4exp.packed_moe import k4_row_order, _moe_chunk_body_k4, _moe_chunk_body_k4rp
dev = torch.device("hpu"); E, H, I = 512, 2560, 640
def planes(name, rows, k):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return [torch.from_numpy(a).to(dev) for a in (rng.integers(0, 256, (E, rows, k // 2), dtype=np.uint8),
            rng.integers(40, 70, (E, rows, k // 16), dtype=np.uint8), (rng.random(E, dtype=np.float32) * 1e-2 + 1e-3).astype(np.float32))]
gate, up, down = planes("gate", I, H), planes("up", I, H), planes("down", H, I)
perm = k4_row_order(I, dev)
gp = [gate[0].index_select(1, perm).contiguous(), gate[1].index_select(1, perm).contiguous(), gate[2]]
upp = [up[0].index_select(1, perm).contiguous(), up[1].index_select(1, perm).contiguous(), up[2]]
c_k4 = torch.compile(_moe_chunk_body_k4, backend="hpu_backend", fullgraph=True, dynamic=False)
c_rp = torch.compile(_moe_chunk_body_k4rp, backend="hpu_backend", fullgraph=True, dynamic=False)
def timeit(fn, reps=20):
    for _ in range(3): fn()
    torch.hpu.synchronize(); t = time.perf_counter()
    for _ in range(reps): fn()
    torch.hpu.synchronize(); return (time.perf_counter() - t) / reps * 1000
rng = np.random.default_rng(3)
for n_c, cap in ((64, 64), (64, 128), (64, 256), (64, 512), (64, 1024)):
    idl = [torch.from_numpy(rng.choice(E, n_c, replace=False).astype(np.int32)).to(dev) for _ in range(8)]
    x = (torch.randn(n_c, cap, H) * 0.5).bfloat16().to(dev)
    a = timeit(lambda: [c_k4(x, i, *gate, *up, *down) for i in idl]) / 8
    b = timeit(lambda: [c_rp(x, i, *gp, *upp, *down) for i in idl]) / 8
    print(f"n={n_c} cap={cap}: k4 {a:6.2f} ms  k4rp {b:6.2f} ms  ({a / b:.2f}x)  per expert k4rp {b / n_c * 1000:6.1f} us", flush=True)
