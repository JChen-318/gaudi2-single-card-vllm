# SPDX-License-Identifier: MIT
"""Run compiler MME descriptors (converted dumps) in the PyTorch/Synapse context.

  --kind qk : C[128, n] = Q[128, 128] @ Kt[128, n]       (capture g-m128n<n>k128)
  --kind pv : C[128, 128] = P[128, k] @ V[k, 128]         (capture g-m128n128k<k>)
--slave keep|drop: keep the compiler's master+slave split, or clear slave (master only).
Correctness: exact match with small integer inputs for qk; for pv the partial output
blocks (128x128 each, consecutive after Cout) are summed and compared with tolerance.
Timing: submit -> all descriptor completions (includes the ~6 us submit/wait floor).
"""
import argparse
import ctypes as C
import json
import time
from pathlib import Path

import torch
import habana_frameworks.torch.core  # noqa: F401
from habana_frameworks.torch.utils.experimental import _data_ptr

import mme_dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--row-repeat2",action="store_true",help="Experimental Conv[1] repeat of pinned M512 capture to M1024")
p.add_argument("--mme-rate",type=lambda s:int(s,0),choices=(0x020202,0x040404,0x080808),default=0x040404,help="Per-descriptor nonzero rate-limiter comparison; no global register writes")
p.add_argument("--kind", choices=("qk", "pv"), required=True)
p.add_argument("--size", type=int, required=True, help="n for qk, k for pv")
p.add_argument("--k", type=int, default=128, help="qk: reduction depth K (capture <tag>-m<rows>n<size>k<K>)")
p.add_argument("--rows", type=int, default=128, help="M (multiple of 128); needs a g-m<rows>... capture")
p.add_argument("--out-hbm", action="store_true", help="Output C in HPU HBM instead of SRAM")
p.add_argument("--no-barrier", action="store_true", help="Experiment: descriptor writes/program end without engine barrier")
p.add_argument("--out-fp32", action="store_true", help="QK: FP32 output (use --fixture-tag g32 capture)")
p.add_argument("--fixture-tag", default="g", help="capture tag prefix (g, np = no SRAM slicing)")
p.add_argument("--bulk", action="store_true", help="WREG_BULK descriptor writes (implied by --no-barrier)")
p.add_argument("--in-hbm", action="store_true", help="Inputs A/B in HPU HBM instead of SRAM")
p.add_argument("--scores-fp16", action="store_true", help="QK: FP16 output, preserves BF16 inputs")
p.add_argument("--pv-owner", type=int, choices=(0,1), help="PV: use capture engine 0 alone, remapped to selected engine")
p.add_argument("--coalesce-qk", action="store_true")
p.add_argument("--batch-input", type=Path, help="Diagnostic: use real Q/K/V, one key half")
p.add_argument("--key-half", type=int, choices=(0,1), default=0)
p.add_argument("--slave", choices=("keep", "drop"), default="keep")
p.add_argument("--repeats", type=int, default=10)
p.add_argument("--store-en1", action="store_true",
               help="Experiment: set header storeEn1 (bit 23 of the 0x30 header word) and give Cout1 its own poisoned SRAM buffer")
p.add_argument("--qk-engine", type=int, choices=(0, 1), help="QK write-wall test: keep only this engine's descriptors")
p.add_argument("--bg-dma", choices=("none", "sram", "hbm"), default="none",
               help="Write-wall test: asynchronous background copies issued just before every MME run")
p.add_argument("--bg-mib", type=int, default=256, help="Background copy volume per run (MiB)")
p.add_argument("--partials", type=int, default=4, help="pv: partial 128x128 blocks to sum")
args = p.parse_args()
if args.rows % 128 or not 128 <= args.rows <= 4096:
    raise ValueError("rows must be a multiple of 128 in [128, 4096]")
root = Path(__file__).resolve().parent
lib = C.CDLL(str(root / "build/libcontext_bridge.so"))
lib.context_info.argtypes = [C.c_uint, C.POINTER(C.c_uint64)]
lib.context_scal_nop.argtypes = [C.c_int]
lib.context_runtime_copy.argtypes = [C.c_uint, C.c_uint64, C.c_uint64, C.c_uint64]
lib.context_set_wreg_bulk.argtypes = [C.c_int]
lib.context_set_mme_nobarrier_allowed.argtypes = [C.c_int]
lib.context_mme_raw_prepare.argtypes = [C.c_uint, C.POINTER(C.c_uint32), C.POINTER(C.c_uint)] + \
    [C.POINTER(C.c_uint64)] * 4
lib.context_mme_raw_run.argtypes = [C.POINTER(C.c_double)]
for name in ("context_info", "context_scal_nop", "context_runtime_copy", "context_mme_raw_prepare",
             "context_mme_raw_run"):
    getattr(lib, name).restype = C.c_int

capture_rows=512 if args.row_repeat2 else args.rows
if args.row_repeat2 and args.rows!=1024:raise ValueError("row repeat2 requires rows=1024")
if args.k != 128 and (args.kind != "qk" or args.batch_input or args.coalesce_qk or args.row_repeat2):
    raise ValueError("--k other than 128 is plain QK only")
shape = f"{args.fixture_tag}-m{capture_rows}n{args.size}k{args.k}" if args.kind == "qk" else f"{args.fixture_tag}-m{capture_rows}n128k{args.size}"
descs = mme_dump.load_set(root / "fixtures" / shape / "mme_debug", keep_slave=args.slave == "keep")
if not descs:
    raise ValueError(f"Missing descriptor capture: {shape}")
if args.scores_fp16:
    if args.kind != "qk":
        raise ValueError("FP16 output is only tested for QK")
    for _, _, words in descs:
        i = (0x30 - 8) // 4
        assert words[i] >> 28 == 9 and (words[i] >> 24) & 15 == 9
        words[i] = words[i] & 0x0FFFFFFF | 8 << 28
if args.coalesce_qk:
    if args.kind != "qk": raise ValueError("Coalescing is QK only")
    descs = mme_dump.coalesce_qk(descs)
if args.row_repeat2:
    descs=mme_dump.repeat_rows_512(descs)
if args.qk_engine is not None:
    if args.kind != "qk": raise ValueError("--qk-engine is QK only")
    descs = [x for x in descs if x[1] == args.qk_engine]
    assert descs, "no descriptor for that engine"
if args.pv_owner is not None:
    if args.kind != "pv": raise ValueError("PV owner is PV only")
    descs = [(act,args.pv_owner,w) for act,e,w in descs if e == 0]
for _,_,words in descs:
    words[(0x29c-8)//4]=args.mme_rate
if args.store_en1:
    for _, _, words in descs:
        h = (0x30 - 8) // 4
        assert (words[h] >> 22) & 1 == 1 and (words[h] >> 23) & 1 == 0, hex(words[h])
        words[h] |= 1 << 23
output_dtype = torch.float32 if args.out_fp32 else (torch.float16 if args.scores_fp16 else torch.bfloat16)
out_bytes = 4 if args.out_fp32 else 2
if args.out_fp32 and args.scores_fp16:
    raise ValueError("--out-fp32 and --scores-fp16 are exclusive")
print("DESCRIPTORS " + json.dumps([(a, d) for a, d, _ in descs]), flush=True)

torch.manual_seed(3)
if args.kind == "qk":
    a = torch.randint(-2, 3, (args.rows, args.k)).bfloat16()
    b = torch.randint(-2, 3, (args.k, args.size)).bfloat16()
    out_elems = args.rows * args.size
else:
    a = torch.randint(0, 3, (args.rows, args.size)).bfloat16()
    b = torch.randint(-1, 2, (args.size, 128)).bfloat16()
    out_elems = args.rows * 128 * args.partials
if args.batch_input:
    import struct
    blob = bytearray(args.batch_input.read_bytes())
    magic, keys, valid, dim, tiles, rows, _ = struct.unpack_from("<8s6I", blob)
    assert magic == b"G2BATCH1" and dim == 128 and args.rows <= rows
    offset, tensors = 32, []
    for shape in ((rows, 128), (keys, 128), (keys, 128)):
        count = shape[0] * shape[1]
        tensors.append(torch.frombuffer(blob, dtype=torch.bfloat16, count=count, offset=offset).clone().reshape(shape))
        offset += count * 2
    q, k, v = tensors
    first = (keys // 128 + 1) // 2 * 128
    lo, hi = (0, first) if args.key_half == 0 else (first, keys)
    assert hi-lo == args.size
    if args.kind == "qk":
        a, b = q[:args.rows], k[lo:hi].T.contiguous()
    else:
        scores = (q[:args.rows].float() @ k[lo:hi].float().T).half().float()
        valid_local = min(valid,hi)-lo
        m = scores[:,:valid_local].max(-1, keepdim=True).values
        a = torch.exp(scores-m).bfloat16(); a[:,valid_local:] = 0
        b = v[lo:hi].contiguous()
reference = a.float() @ b.float()

ah, bh = a.to("hpu"), b.to("hpu")  # acquire the device before borrowing its context
torch.hpu.synchronize()
info = (C.c_uint64 * 5)()
assert lib.context_info(0, info) == 0
assert lib.context_scal_nop(int(info[0])) == 0
base = info[1] + 0x200000
sa = base
sb = (sa + a.numel() * 2 + 255) & ~255
sc = (sb + b.numel() * 2 + 255) & ~255
# HBM placement uses owned contiguous HPU allocations (no slice data_ptr).
hbm_out = torch.empty((out_elems + 128,), dtype=output_dtype, device="hpu") if args.out_hbm else None
if args.in_hbm:
    sa, sb = _data_ptr(ah), _data_ptr(bh)
    sc = base
if args.out_hbm:
    sc = (_data_ptr(hbm_out) + 255) & ~255
end = (0 if args.out_hbm else sc + out_elems * out_bytes)
sc1 = 0
if args.store_en1:
    assert not args.out_hbm
    sc1 = (end + 4096 + 255) & ~255
    end = sc1 + out_elems * out_bytes
assert end + 4096 <= info[1] + info[2], "SRAM too small"


def copy(src, dst, size):
    assert lib.context_runtime_copy(0, src, dst, size) == 0


if not args.in_hbm:
    copy(_data_ptr(ah), sa, a.numel() * 2)
    copy(_data_ptr(bh), sb, b.numel() * 2)
n = len(descs)
words = (C.c_uint32 * (182 * n))(*[w for _, _, ws in descs for w in ws])
engine = (C.c_uint * n)(*[d for _, d, _ in descs])
A = (C.c_uint64 * n)(*[sa] * n)
B = (C.c_uint64 * n)(*[sb] * n)
C0 = (C.c_uint64 * n)(*[sc] * n)
C1 = (C.c_uint64 * n)(*[sc1] * n)
if args.bulk and not args.no_barrier:
    assert lib.context_set_wreg_bulk(1) == 0
if args.no_barrier:
    # Bulk writes are required for the no-barrier WREG_BULK form.
    assert lib.context_set_wreg_bulk(1) == 0 and lib.context_set_mme_nobarrier_allowed(1) == 0
status = lib.context_mme_raw_prepare(n, words, engine, A, B, C0, C1)
assert status == 0, f"prepare {status}"
poison = torch.full((out_elems,), -7.0, dtype=output_dtype, device="hpu")
out = torch.empty((out_elems,), dtype=output_dtype, device="hpu")
check = torch.randn(64, 64).to("hpu")
before = (check @ check).cpu()
u64 = C.c_uint64
enq = lib.context_batch_dma_enqueue; enq.argtypes = [C.c_uint] + [C.POINTER(u64)] * 3; enq.restype = C.c_int
dsync = lib.context_batch_dma_sync; dsync.argtypes = []; dsync.restype = C.c_int
bg_batches, bg_alone_us = [], []
if args.bg_dma != "none":
    chunk = 4 << 20
    src = torch.empty(64 << 20, dtype=torch.uint8, device="hpu")
    if args.bg_dma == "sram":
        lo = (end + 4096 + 255) & ~255
        hi = info[1] + info[2] - 4096
        assert hi - lo >= 4 * chunk, "not enough free SRAM for background copies"
        dsts = [lo + i * chunk for i in range((hi - lo) // chunk)][:4]
    else:
        dst_t = torch.empty(64 << 20, dtype=torch.uint8, device="hpu")
        dsts = [_data_ptr(dst_t) + i * chunk for i in range(16)]
    torch.hpu.synchronize()
    entries = [(_data_ptr(src) + (i % 16) * chunk, dsts[i % len(dsts)], chunk) for i in range(args.bg_mib // 4)]
    for i in range(0, len(entries), 8):
        e = entries[i:i + 8]; n_ = len(e)
        bg_batches.append((n_, (u64 * n_)(*[x[0] for x in e]), (u64 * n_)(*[x[1] for x in e]), (u64 * n_)(*[x[2] for x in e])))
    for _ in range(3):  # background copy alone: its window must cover the MME run
        t = time.perf_counter()
        for b in bg_batches: assert enq(*b) == 0
        assert dsync() == 0
        bg_alone_us.append((time.perf_counter() - t) * 1e6)
times = []
for it in range(args.repeats):
    copy(_data_ptr(poison), sc, out_elems * out_bytes)
    if sc1: copy(_data_ptr(poison), sc1, out_elems * out_bytes)
    for b in bg_batches: assert enq(*b) == 0
    us = C.c_double()
    status = lib.context_mme_raw_run(C.byref(us))
    assert status == 0, f"run {it}: {status}"
    if bg_batches: assert dsync() == 0
    times.append(us.value)
copy(sc, _data_ptr(out), out_elems * out_bytes)
got = out.cpu().float()
got1 = None
if sc1:
    out1 = torch.empty((out_elems,), dtype=output_dtype, device="hpu")
    copy(sc1, _data_ptr(out1), out_elems * out_bytes)
    got1 = out1.cpu().float()
result = {"kind": args.kind, "size": args.size, "rows": args.rows, "k": args.k, "out_hbm": args.out_hbm, "in_hbm": args.in_hbm, "fixture_tag": args.fixture_tag, "out_fp32": args.out_fp32, "no_barrier": args.no_barrier, "bulk": args.bulk or args.no_barrier, "scores_fp16": args.scores_fp16, "slave": args.slave, "coalesce_qk": args.coalesce_qk, "pv_owner": args.pv_owner, "descriptors": n,
          "us": times, "median_us": sorted(times)[len(times) // 2]}
if args.kind == "qk":
    got = got.reshape(args.rows, args.size)
    if args.qk_engine is not None or args.store_en1:
        written = got != -7
        ref_c = reference.to(output_dtype).float()
        result["written_fraction"] = float(written.float().mean())
        result["exact"] = bool(torch.equal(got[written], ref_c[written])) and result["written_fraction"] > 0.4
    else:
        result["exact"] = bool(torch.equal(got, reference.to(output_dtype).float()))
    result["real_input"] = bool(args.batch_input)
    result["relative_error"] = float((got-reference.to(output_dtype).float()).norm()/reference.norm())
    result["max_absolute_error"] = float((got-reference.to(output_dtype).float()).abs().max())
    result["mismatch_fraction"] = float((got != reference.to(output_dtype).float()).float().mean())
else:
    blocks = got.reshape(args.partials, args.rows, 128)
    poisoned = [bool((blk == -7).all()) for blk in blocks]
    total = blocks[[i for i, q in enumerate(poisoned) if not q]].sum(0) if not all(poisoned) else blocks[0]
    rel = ((total - reference).norm() / reference.norm()).item()
    result.update({"poisoned_blocks": poisoned, "sum_rel_error": rel,
                   "block0_rel_error": ((blocks[0] - reference).norm() / reference.norm()).item()})
result["mme_rate"] = args.mme_rate
result["qk_engine"] = args.qk_engine
result["store_en1"] = args.store_en1
if got1 is not None and args.kind == "qk":
    ref_c = reference.to(output_dtype).float().reshape(-1)
    g0, g1 = got.reshape(-1), got1.reshape(-1)
    w0, w1 = g0 != -7, g1 != -7
    result["cout0_written_fraction"] = float(w0.float().mean())
    result["cout1_written_fraction"] = float(w1.float().mean())
    result["cout1_matches_reference_where_written"] = bool(torch.equal(g1[w1], ref_c[w1])) if bool(w1.any()) else None
    result["union_written_fraction"] = float((w0 | w1).float().mean())
    result["cout1_equals_cout0_full"] = bool(torch.equal(g0, g1))
result["bg_dma"] = args.bg_dma; result["bg_mib"] = args.bg_mib if bg_batches else 0
result["bg_alone_us"] = bg_alone_us
result["row_repeat2"] = args.row_repeat2
result["torch_after"] = bool(torch.equal((check @ check).cpu(), before))
assert result["torch_after"]
if args.kind == "qk" and args.batch_input:
    assert result["relative_error"] < 0.001, result
else:
    assert result.get("exact", result.get("sum_rel_error", 1) < 0.01), result
print("RESULT_MME_RAW " + json.dumps(result), flush=True)
