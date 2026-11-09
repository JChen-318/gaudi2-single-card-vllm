# SPDX-License-Identifier: MIT
"""QK -> PV overlap on the same MMEs (2026-11-09).

Production geometry quarter: QK C[512, 9472] = Q[512, 128] @ Kt[128, 9472] (FP16 scores,
capture g-m512n9472k128, 6 descriptors) and PV O = P[512, 9472] @ V[9472, 128] (capture
g-m512n128k9472, 2 descriptors). Independent buffers, so any overlap is legal here.

--mode qk | pv | qkpv (QK descriptors then PV on each engine) | pvqk (PV first).
--no-barrier commits descriptors back to back without engine barriers (as wide-relaxed).
Serial hardware: T(qkpv) ~ T(qk) + T(pv) - floor. Overlap: closer to max(T(qk), T(pv)).
Timing is CPU steady_clock around submit + host fence wait (median), not device cycles.
"""
import argparse, json
import ctypes as C
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
from habana_frameworks.torch.utils.experimental import _data_ptr

import mme_dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mode", choices=("qk", "pv", "qkpv", "pvqk"), required=True)
p.add_argument("--no-barrier", action="store_true")
p.add_argument("--repeats", type=int, default=20)
p.add_argument("--partials", type=int, default=4)
args = p.parse_args()
ROWS, N = 512, 9472
root = Path(__file__).resolve().parent
lib = C.CDLL(str(root / "build/libcontext_bridge.so"))
lib.context_info.argtypes = [C.c_uint, C.POINTER(C.c_uint64)]
lib.context_scal_nop.argtypes = [C.c_int]
lib.context_runtime_copy.argtypes = [C.c_uint, C.c_uint64, C.c_uint64, C.c_uint64]
lib.context_set_wreg_bulk.argtypes = [C.c_int]
lib.context_set_mme_nobarrier_allowed.argtypes = [C.c_int]
lib.context_mme_raw_prepare.argtypes = [C.c_uint, C.POINTER(C.c_uint32), C.POINTER(C.c_uint)] + [C.POINTER(C.c_uint64)] * 4
lib.context_mme_raw_run.argtypes = [C.POINTER(C.c_double)]
for name in ("context_info", "context_scal_nop", "context_runtime_copy", "context_mme_raw_prepare", "context_mme_raw_run",
             "context_set_wreg_bulk", "context_set_mme_nobarrier_allowed"):
    getattr(lib, name).restype = C.c_int

qk_descs = mme_dump.load_set(root / "fixtures" / f"g-m{ROWS}n{N}k128" / "mme_debug", keep_slave=True)
pv_descs = mme_dump.load_set(root / "fixtures" / f"g-m{ROWS}n128k{N}" / "mme_debug", keep_slave=True)
assert len(qk_descs) == 6 and len(pv_descs) == 2
for _, _, w in qk_descs:  # FP16 scores, BF16 inputs (as --scores-fp16)
    i = (0x30 - 8) // 4
    assert w[i] >> 28 == 9 and (w[i] >> 24) & 15 == 9
    w[i] = w[i] & 0x0FFFFFFF | 8 << 28
for _, _, w in qk_descs + pv_descs:
    w[(0x29c - 8) // 4] = 0x040404

torch.manual_seed(3)
q = torch.randint(-2, 3, (ROWS, 128)).bfloat16(); kt = torch.randint(-2, 3, (128, N)).bfloat16()
pm = torch.randint(0, 3, (ROWS, N)).bfloat16(); v = torch.randint(-1, 2, (N, 128)).bfloat16()
ref_qk = (q.float() @ kt.float()).half().float()
ref_pv = pm.float() @ v.float()
qk_out = ROWS * N; pv_out = ROWS * 128 * args.partials

tensors = [t.to("hpu") for t in (q, kt, pm, v)]
torch.hpu.synchronize()
info = (C.c_uint64 * 5)()
assert lib.context_info(0, info) == 0 and lib.context_scal_nop(int(info[0])) == 0
cur = info[1] + 0x200000
def alloc(nbytes):
    global cur
    addr = (cur + 255) & ~255; cur = addr + nbytes; return addr
sq, sk, sp, sv = (alloc(t.numel() * 2) for t in (q, kt, pm, v))
sc_qk = alloc(qk_out * 2); sc_pv = alloc(pv_out * 2)
assert cur + 4096 <= info[1] + info[2], "SRAM too small"
def copy(src, dst, size): assert lib.context_runtime_copy(0, src, dst, size) == 0
for t, s in zip(tensors, (sq, sk, sp, sv)): copy(_data_ptr(t), s, t.numel() * 2)

sets = {"qk": [("qk", d) for d in qk_descs], "pv": [("pv", d) for d in pv_descs]}
order = {"qk": ["qk"], "pv": ["pv"], "qkpv": ["qk", "pv"], "pvqk": ["pv", "qk"]}[args.mode]
descs = [x for k in order for x in sets[k]]
n = len(descs); assert n <= 8
addr = {"qk": (sq, sk, sc_qk), "pv": (sp, sv, sc_pv)}
words = (C.c_uint32 * (182 * n))(*[w for _, (_, _, ws) in descs for w in ws])
engine = (C.c_uint * n)(*[d for _, (_, d, _) in descs])
A = (C.c_uint64 * n)(*[addr[k][0] for k, _ in descs])
B = (C.c_uint64 * n)(*[addr[k][1] for k, _ in descs])
C0 = (C.c_uint64 * n)(*[addr[k][2] for k, _ in descs])
C1 = (C.c_uint64 * n)(*[0] * n)
assert lib.context_set_wreg_bulk(1) == 0
if args.no_barrier: assert lib.context_set_mme_nobarrier_allowed(1) == 0
assert lib.context_mme_raw_prepare(n, words, engine, A, B, C0, C1) == 0

pq = torch.full((qk_out,), -7.0, dtype=torch.float16, device="hpu")
pp = torch.full((pv_out,), -7.0, dtype=torch.bfloat16, device="hpu")
check = torch.randn(64, 64).to("hpu"); before = (check @ check).cpu()
times = []
for it in range(args.repeats):
    copy(_data_ptr(pq), sc_qk, qk_out * 2); copy(_data_ptr(pp), sc_pv, pv_out * 2)
    us = C.c_double()
    assert lib.context_mme_raw_run(C.byref(us)) == 0, f"run {it}"
    times.append(us.value)
res = {"mode": args.mode, "no_barrier": args.no_barrier, "descriptors": n,
       "order": [(k, a, d) for k, (a, d, _) in descs], "us": times, "median_us": sorted(times)[len(times) // 2]}
if "qk" in order:
    o = torch.empty((qk_out,), dtype=torch.float16, device="hpu"); copy(sc_qk, _data_ptr(o), qk_out * 2)
    res["qk_exact"] = bool(torch.equal(o.cpu().float().reshape(ROWS, N), ref_qk))
if "pv" in order:
    o = torch.empty((pv_out,), dtype=torch.bfloat16, device="hpu"); copy(sc_pv, _data_ptr(o), pv_out * 2)
    blocks = o.cpu().float().reshape(args.partials, ROWS, 128)
    live = [i for i in range(args.partials) if not bool((blocks[i] == -7).all())]
    res["pv_live_blocks"] = live
    res["pv_rel_error"] = float(((blocks[live].sum(0) - ref_pv).norm() / ref_pv.norm()).item()) if live else 1.0
res["torch_after"] = bool(torch.equal((check @ check).cpu(), before))
ok = res["torch_after"] and res.get("qk_exact", True) and res.get("pv_rel_error", 0) < 0.01
print("RESULT_MME_OVERLAP " + json.dumps(res), flush=True)
assert ok, res
