"""Read-only borrowed FD / tensor VA / SCAL inspection, then framework checks.

The earlier direct-driver DMA experiment is disabled in the normal build.
It timed out under an active Synapse instance and caused an automatic reset.
"""
import ctypes
import json
import sys
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
from habana_frameworks.torch.utils.experimental import _data_ptr

root = Path(__file__).resolve().parent
lib = ctypes.CDLL(str(root / "build/libcontext_bridge.so"))
lib.context_info.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64)]
lib.context_scal_info.argtypes = [ctypes.c_int]
lib.context_scal_nop.argtypes = [ctypes.c_int]
lib.context_scal_mme.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
lib.context_scal_mme_ecb.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
lib.context_scal_fence_test.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
lib.context_scal_tpc_relu.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p]
lib.context_scal_tpc_fence_test.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p]
lib.context_runtime_copy.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
for name in ["context_info", "context_scal_info", "context_scal_nop", "context_scal_mme", "context_scal_mme_ecb", "context_scal_fence_test"]:
    getattr(lib, name).restype = ctypes.c_int
lib.context_runtime_copy.restype = ctypes.c_int
lib.context_scal_tpc_relu.restype = ctypes.c_int
lib.context_scal_tpc_fence_test.restype = ctypes.c_int

torch.manual_seed(20261108)
x_cpu = torch.arange(1024, dtype=torch.float32)
x = x_cpu.to("hpu")
y = torch.full_like(x, -1)
check = torch.randn(128, 128)
check_hpu = check.to("hpu")
mm = torch.compile(lambda a: a @ a, backend="hpu_backend", fullgraph=True)
before = mm(check_hpu).cpu()
torch.hpu.synchronize()
info = (ctypes.c_uint64 * 5)()
assert lib.context_info(0, info) == 0
fd = int(info[0])
print("CONTEXT " + json.dumps({"fd": fd, "sram": hex(info[1]), "sram_bytes": info[2],
                              "x": hex(_data_ptr(x)), "y": hex(_data_ptr(y))}), flush=True)
assert _data_ptr(x) >= 0x1000000000000000, "HPU data_ptr is not the expected device VA"
assert lib.context_scal_info(fd) == 0
runtime_results = {}
if any(flag in sys.argv for flag in ("--scal-nop", "--scal-mme", "--scal-mme-ecb", "--scal-tpc", "--scal-fence", "--scal-tpc-fence")):
    status = lib.context_scal_nop(fd)
    runtime_results["own_scal_nop"] = status
    assert status == 0, f"SCAL NOP returned {status}"
if any(flag in sys.argv for flag in ("--scal-mme", "--scal-mme-ecb", "--scal-fence")):
    for iteration in range(3):
        a_cpu = torch.randint(-4, 5, (128, 128)).bfloat16()
        b_cpu = torch.randint(-4, 5, (128, 128)).bfloat16()
        a, b = a_cpu.to("hpu"), b_cpu.to("hpu")
        output = torch.full_like(a, -1)
        torch.hpu.synchronize()
        launch = lib.context_scal_fence_test if "--scal-fence" in sys.argv else (
            lib.context_scal_mme_ecb if "--scal-mme-ecb" in sys.argv else lib.context_scal_mme)
        status = launch(_data_ptr(a), _data_ptr(b), _data_ptr(output))
        runtime_results["own_scal_mme"] = status
        assert status == 0, f"SCAL MME returned {status}"
        expected = (a_cpu.float() @ b_cpu.float()).bfloat16()
        assert torch.equal(output.cpu(), expected), "SCAL MME result mismatch"
        assert torch.equal(mm(check_hpu).cpu(), before), "Framework GEMM changed after ECB"
        print("RESULT_SCAL_MME " + json.dumps({"iteration": iteration, "elements": 16384,
                                              "bitwise_equal": True}), flush=True)
if "--scal-tpc" in sys.argv or "--scal-tpc-fence" in sys.argv:
    for iteration in range(3):
        a_cpu = torch.randint(-4, 5, (128, 128)).bfloat16()
        a = a_cpu.to("hpu")
        output = torch.full_like(a, -1)
        torch.hpu.synchronize()
        launch = lib.context_scal_tpc_fence_test if "--scal-tpc-fence" in sys.argv else lib.context_scal_tpc_relu
        status = launch(_data_ptr(a), _data_ptr(output), str(root / "build/relu.bin").encode())
        runtime_results["own_scal_tpc"] = status
        assert status == 0, f"SCAL TPC returned {status}"
        assert torch.equal(output.cpu(), a_cpu.clamp_min(0)), "SCAL TPC result mismatch"
        assert torch.equal(mm(check_hpu).cpu(), before), "Framework GEMM changed after TPC ECB"
        print("RESULT_SCAL_TPC " + json.dumps({"iteration": iteration, "elements": 16384,
                                              "bitwise_equal": True}), flush=True)
if "--runtime-copy" in sys.argv:
    # This probe runs alone in a temporary container. No graph is executing.
    scratch = info[1] + info[2] - 65536
    for name, src, dst in [("runtime_hbm_copy", _data_ptr(x), _data_ptr(y)),
                           ("runtime_to_sram", _data_ptr(x), scratch),
                           ("runtime_from_sram", scratch, _data_ptr(y))]:
        if name == "runtime_from_sram":
            y.fill_(-1)
            torch.hpu.synchronize()
        status = lib.context_runtime_copy(0, src, dst, 4096)
        runtime_results[name] = status
        print("RUNTIME_COPY " + json.dumps({"name": name, "status": status}), flush=True)
        if status:
            break
        if dst == _data_ptr(y):
            assert torch.equal(y.cpu(), x_cpu), name + " data mismatch"
after = mm(check_hpu).cpu()
assert torch.equal(after, before), "Framework GEMM changed after read-only inspection"
assert torch.equal((x + 2).cpu(), x_cpu + 2), "Framework vector result changed"
print("RESULT " + json.dumps({"scal_read_only_inspection": True, "torch_after": True,
                             "raw_queue_interop": False, **runtime_results}), flush=True)
