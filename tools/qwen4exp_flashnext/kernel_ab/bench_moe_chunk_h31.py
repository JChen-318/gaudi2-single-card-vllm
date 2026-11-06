"""Prefill MoE expert-chunk body: eager vs one compiled graph (H31).

Body per chunk (64 experts, cap rows each): dequant gate/up/down (TPC) + 3 bmm (MME) + silu*mul.
Eager runs it op by op on one stream; a compiled graph lets the graph compiler overlap the TPC
dequant of one projection with the MME bmm of another and keep the small planes in SRAM.
"""
import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
REPO = Path(__file__).resolve().parents[3]
os.environ["GC_KERNEL_PATH"] = str(REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so") + \
    ":/usr/lib/habanalabs/libtpc_kernels.so"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402,F401

torch.ops.load_library(str(next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))))
ops = torch.ops.custom_op
dev = torch.device("hpu")
E, H, I = 512, 2560, 640


def planes(name, rows, k):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return [torch.from_numpy(a).to(dev) for a in (
        rng.integers(0, 256, (E, rows, k // 2), dtype=np.uint8), rng.integers(40, 70, (E, rows, k // 16), dtype=np.uint8),
        (rng.random(E, dtype=np.float32) * 1e-2 + 1e-3).astype(np.float32))]


gate, up, down = planes("gate", I, H), planes("up", I, H), planes("down", H, I)


def body(bucketed, ids):
    gate_w = ops.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(ids, *gate)
    up_w = ops.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(ids, *up)
    gate_out = torch.bmm(bucketed, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed, up_w.transpose(1, 2))
    act = F.silu(gate_out) * up_out
    down_w = ops.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(ids, *down)
    return torch.bmm(act, down_w.transpose(1, 2))


compiled = torch.compile(body, backend="hpu_backend", fullgraph=True, dynamic=False)


def timeit(fn, reps=30):
    for _ in range(3):
        fn()
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.hpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


rng = np.random.default_rng(3)
for n_c, cap in ((64, 64), (64, 256), (64, 1024), (32, 128)):
    ids_list = [torch.from_numpy(rng.choice(E, n_c, replace=False).astype(np.int32)).to(dev) for _ in range(8)]
    x = (torch.randn(n_c, cap, H) * 0.5).bfloat16().to(dev)
    seq = lambda f: [f(x, ids) for ids in ids_list]
    te = timeit(lambda: seq(body)) / len(ids_list)
    tg = timeit(lambda: seq(compiled)) / len(ids_list)
    a = body(x, ids_list[0]).float().cpu()
    b = compiled(x, ids_list[0]).float().cpu()
    rel = float((a - b).abs().max() / a.abs().max())
    print(f"n_c={n_c:3d} cap={cap:5d}: eager {te:7.2f} ms  graph {tg:7.2f} ms  ({te / tg:.2f}x)  max rel diff {rel:.2e}",
          flush=True)


def k4_layout(x):
    lead, width = x.shape[:-1], x.shape[-1]
    full = width // 512 * 512
    head = x[..., :full].reshape(*lead, full // 512, 128, 4).transpose(-1, -2).reshape(*lead, full)
    return head.contiguous() if full == width else torch.cat([head, x[..., full:]], dim=-1)


def body_k4(bucketed_k4, ids):
    dq = ops.custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2
    gate_w = dq(ids, *gate)
    up_w = dq(ids, *up)
    gate_out = torch.bmm(bucketed_k4, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed_k4, up_w.transpose(1, 2))
    act = k4_layout(F.silu(gate_out) * up_out)
    down_w = dq(ids, *down)
    return torch.bmm(act, down_w.transpose(1, 2))


compiled_k4 = torch.compile(body_k4, backend="hpu_backend", fullgraph=True, dynamic=False)
print("--- k4 dequant inside the graph ---", flush=True)
# value check of the layout: dequant_k4 equals the natural dequant permuted along K
ids0 = torch.arange(4, dtype=torch.int32, device=dev)
for name, pl in (("gate", gate), ("down", down)):
    nat = ops.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2(ids0, *pl).cpu()
    k4v = ops.custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2(ids0, *pl).cpu()
    print(f"{name}: dequant_k4 == k4_layout(dequant): {torch.equal(k4_layout(nat), k4v)}", flush=True)
for n_c, cap in ((64, 64), (64, 256), (64, 1024)):
    ids_list = [torch.from_numpy(rng.choice(E, n_c, replace=False).astype(np.int32)).to(dev) for _ in range(8)]
    x = (torch.randn(n_c, cap, H) * 0.5).bfloat16().to(dev)
    xk = k4_layout(x)
    tg = timeit(lambda: [compiled(x, i) for i in ids_list]) / len(ids_list)
    tk = timeit(lambda: [compiled_k4(xk, i) for i in ids_list]) / len(ids_list)
    a = compiled(x, ids_list[0]).float().cpu()
    b = compiled_k4(xk, ids_list[0]).float().cpu()
    rel = float((a - b).abs().max() / a.abs().max())
    differ = float((a != b).float().mean())
    print(f"n_c={n_c} cap={cap}: graph {tg:6.2f} ms  graph+k4 {tk:6.2f} ms ({tg / tk:.2f}x)  "
          f"max rel diff {rel:.2e}, elements differing {differ * 100:.2f}%", flush=True)


def body_k4b(bucketed_k4, ids):
    dq4 = ops.custom_qwen4exp_nvfp4_indexed_dequant_k4_bf16_gaudi2
    dq = ops.custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2
    gate_w = dq4(ids, *gate)
    up_w = dq4(ids, *up)
    gate_out = torch.bmm(bucketed_k4, gate_w.transpose(1, 2))
    up_out = torch.bmm(bucketed_k4, up_w.transpose(1, 2))
    act = F.silu(gate_out) * up_out
    down_w = dq(ids, *down)
    return torch.bmm(act, down_w.transpose(1, 2))


compiled_k4b = torch.compile(body_k4b, backend="hpu_backend", fullgraph=True, dynamic=False)
print("--- k4 gate/up only ---", flush=True)
for n_c, cap in ((64, 64), (64, 256), (64, 1024), (16, 128)):
    ids_list = [torch.from_numpy(rng.choice(E, n_c, replace=False).astype(np.int32)).to(dev) for _ in range(8)]
    x = (torch.randn(n_c, cap, H) * 0.5).bfloat16().to(dev)
    xk = k4_layout(x)
    tg = timeit(lambda: [compiled(x, i) for i in ids_list]) / len(ids_list)
    tk = timeit(lambda: [compiled_k4b(xk, i) for i in ids_list]) / len(ids_list)
    a = compiled(x, ids_list[0]).float().cpu()
    b = compiled_k4b(xk, ids_list[0]).float().cpu()
    print(f"n_c={n_c} cap={cap}: graph {tg:6.2f} ms  graph+k4(gate/up) {tk:6.2f} ms ({tg / tk:.2f}x)  "
          f"max rel diff {float((a - b).abs().max() / a.abs().max()):.2e}, differing {float((a != b).float().mean()) * 100:.2f}%",
          flush=True)
