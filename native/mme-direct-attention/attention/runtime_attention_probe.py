# SPDX-License-Identifier: MIT
"""Quiesced, isolated-process proof of SRAM attention inside the HPU runtime.

Input/output copies use Synapse streams. Custom MME/TPC jobs use a dedicated
SCAL stream and HBM-resident QMAN programs. By default CPU synchronization
separates stages; --pipeline uses cached commands and device-side fences.
Both paths require an isolated process and quiesced framework streams. Replay
timing excludes input/output copies and preparation; it is not layer throughput.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import struct
import time

import torch
import habana_frameworks.torch.core  # noqa: F401
from habana_frameworks.torch.utils.experimental import _data_ptr

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path)
p.add_argument("--keys", type=int, default=128)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--pipeline", action="store_true", help="Cached ECB with device-side stage fences")
p.add_argument("--tpcs", type=int, default=24)
p.add_argument("--repeats", type=int, default=3)
MODES = ("full", "qk", "qk-exp", "qk-exp-pv", "exp", "pv", "norm", "lbw", "floor")
p.add_argument("--mode", choices=MODES, default="full",
               help="Pipeline timing ablation; only full validates the attention result")
p.add_argument("--exp-kernel", default="exp_sum", choices=("exp_sum", "exp_sum_g2", "exp_sum_g4", "exp_sum_g4b3", "exp_sum_g4b4", "exp_lut", "exp_lut2", "noop4", "exp_lut_split", "exp_lut_half", "exp_lut_half_m", "exp_lut_half_ms8", "exp_lut_half_logq", "exp_lut_half_logq_ms8", "exp_lut_half_logq_b80_ms8"))
p.add_argument("--lut", action="store_true", help="Upload Gaudi2 special-function tables and set TPC LUT bases")
p.add_argument("--lut-mask", type=int, default=2, help="Bit t sets LUT_FUNC{256,128,64,32} base t (default 2: 128 only)")
p.add_argument("--dual-qk", action="store_true", help="Split QK columns across DCORE0 and DCORE2 MME masters")
p.add_argument("--wreg-bulk", action="store_true", help="Upload descriptors with WREG_BULK packets")
p.add_argument("--compiler-qk", action="store_true", help="Full-width compiler master/slave QK with strided split-PV views")
p.add_argument("--alternate-query", action="store_true", help="Correctness: odd blocks use reversed query rows, with a separate reference")
p.add_argument("--blocks", type=int, default=0,
               help="Serial chain of N blocks in one submission (context_attention_run_n); reports per-block time")
p.add_argument("--pipelined", type=int, default=0,
               help="N software-pipelined blocks (double-buffered S/sums/output, needs --dual-qk) in one submission")
p.add_argument("--defer-norm", action="store_true",
               help="With --pipelined: normalize only the last block of each parity (per-head normalization)")
p.add_argument("--tpc-prefence", action="store_true",
               help="With --pipelined: write TPC descriptors before the fence (no engine barrier)")
p.add_argument("--split-pv", action="store_true",
               help="With --pipelined: key halves per MME for QK and PV (exp_lut_split + normalize2)")
p.add_argument("--half-max", action="store_true",
               help="With --split-pv: per-half row maxima (exp_lut_half + normalize3), 256 work units")
p.add_argument("--tpc-rate", default="8,6", help="TPC kernel_config rd,wr rate-limit reset tokens (default 8,6)")
p.add_argument("--rearm-only", action="store_true", help="Write monitor config once; later replays only arm")
args = p.parse_args()
torch.set_num_threads(8)
torch.manual_seed(args.seed)
root = Path(__file__).resolve().parent
lib = C.CDLL(str(root / "build/libcontext_bridge.so"))
lib.context_info.argtypes = [C.c_uint, C.POINTER(C.c_uint64)]
lib.context_scal_nop.argtypes = [C.c_int]
lib.context_runtime_copy.argtypes = [C.c_uint, C.c_uint64, C.c_uint64, C.c_uint64]
lib.context_scal_mme_ex.argtypes = [C.c_uint64] * 3 + [C.c_uint] * 3
lib.context_scal_tpc.argtypes = [C.POINTER(C.c_uint64)] + [C.POINTER(C.c_uint)] * 3 + [C.c_uint, C.c_char_p, C.c_uint]
lib.context_attention_prepare.argtypes = [C.c_uint64] + [C.c_uint] * 3 + [C.c_char_p] * 2
lib.context_attention_run.argtypes = []
lib.context_attention_prepare.restype = lib.context_attention_run.restype = C.c_int
lib.context_attention_prepare_mode.argtypes = [C.c_uint64] + [C.c_uint] * 3 + [C.c_char_p] * 2 + [C.c_uint]
lib.context_attention_prepare_mode.restype = C.c_int
lib.context_attention_last_timing.argtypes = [C.POINTER(C.c_double), C.POINTER(C.c_uint)]
lib.context_attention_last_timing.restype = C.c_int
full = args.mode == "full" and args.exp_kernel != "noop4"  # noop4: timing only, no numeric checks
lib.context_attention_set_rearm_only.argtypes = [C.c_int]
lib.context_attention_set_rearm_only.restype = C.c_int
assert lib.context_attention_set_rearm_only(int(args.rearm_only)) == 0
lib.context_attention_set_dual_qk.argtypes = [C.c_int]
lib.context_attention_set_dual_qk.restype = C.c_int
assert lib.context_attention_set_dual_qk(int(args.dual_qk)) == 0
lib.context_attention_set_chain.argtypes = [C.c_int]
lib.context_attention_set_chain.restype = C.c_int
assert lib.context_attention_set_chain(int(args.blocks > 0)) == 0
lib.context_attention_run_n.argtypes = [C.c_uint]
lib.context_attention_run_n.restype = C.c_int
if args.pipelined and (not args.pipeline or args.mode != "full" or not args.dual_qk or args.blocks or not 2 <= args.pipelined <= 20):
    raise ValueError("--pipelined N needs --pipeline --dual-qk, mode full, 2 <= N <= 20")
if args.blocks and (not args.pipeline or not full or args.blocks > 20):
    raise ValueError("--blocks needs --pipeline, --mode full and N <= 20")
lib.context_set_wreg_bulk.argtypes = [C.c_int]
lib.context_set_wreg_bulk.restype = C.c_int
assert lib.context_set_wreg_bulk(int(args.wreg_bulk)) == 0
lib.context_set_tpc_rate_tokens.argtypes = [C.c_uint, C.c_uint]
lib.context_set_tpc_rate_tokens.restype = C.c_int
assert lib.context_set_tpc_rate_tokens(*map(int, args.tpc_rate.split(","))) == 0
if args.exp_kernel.startswith("exp_lut") and not args.lut:
    raise ValueError("exp_lut requires --lut")
if not full and not args.pipeline:
    raise ValueError("--mode requires --pipeline")
for name in ("context_info", "context_scal_nop", "context_runtime_copy", "context_scal_mme_ex", "context_scal_tpc"):
    getattr(lib, name).restype = C.c_int

keys = valid = args.keys
if args.input:
    data = bytearray(args.input.read_bytes())
    magic, keys, valid, dim, rows = struct.unpack_from("<8s4I", data)
    if magic != b"G2ATTN1\0" or (dim, rows) != (128, 128):
        raise ValueError("Invalid input format")
    offset = 24
    tensors = []
    for shape in ((128, 128), (keys, 128), (keys, 128)):
        count = shape[0] * shape[1]
        tensors.append(torch.frombuffer(data, dtype=torch.bfloat16, count=count, offset=offset).clone().reshape(shape))
        offset += count * 2
    q, k, v = tensors
    reference = torch.frombuffer(data, dtype=torch.float32, count=16384, offset=offset).clone().reshape(128, 128)
else:
    q = (torch.randint(-4, 5, (128, 128)).float() / 16).bfloat16()
    k = (torch.randint(-4, 5, (keys, 128)).float() / 16).bfloat16()
    v = (torch.randint(-4, 5, (keys, 128)).float() / 4).bfloat16()
    reference = torch.softmax(q.float() @ k.float().T, dim=-1) @ v.float()
if not keys or keys % 128 or keys > 37760 or not 0 < valid <= keys:
    raise ValueError("Unsupported key length")
if args.compiler_qk:
    if not args.split_pv or not args.pipelined or not args.wreg_bulk or args.pipelined > 19:
        raise ValueError("--compiler-qk needs --split-pv --pipelined 2..19 --wreg-bulk")
    import mme_dump
    descs = mme_dump.load_set(root / "fixtures" / f"g-m128n{keys}k128" / "mme_debug")
    if not descs or any(d not in (0, 1) for _, d, _ in descs):
        raise ValueError("Missing/unexpected compiler QK capture")
    words = (C.c_uint32 * (182 * len(descs)))(*[w for _, _, ws in descs for w in ws])
    engines = (C.c_uint * len(descs))(*[d for _, d, _ in descs])
    lib.context_attention_set_compiler_qk.argtypes = [C.c_uint, C.POINTER(C.c_uint32), C.POINTER(C.c_uint)]
    lib.context_attention_set_compiler_qk.restype = C.c_int
    assert lib.context_attention_set_compiler_qk(len(descs), words, engines) == 0
    print("COMPILER_QK " + json.dumps({"keys": keys, "descriptors": [(a, d) for a, d, _ in descs],
        "score_row_stride": keys, "monitor_stride": 52}), flush=True)
if not 1 <= args.tpcs <= 24 or not 1 <= args.repeats <= 200:
    raise ValueError("Unsupported TPC/replay count")
bound_cpu = q.float().norm(dim=-1) * k[:valid].float().norm(dim=-1).max()
if not torch.isfinite(bound_cpu).all() or bound_cpu.max() > 80:
    raise ValueError("Use stable-softmax fallback for this input")
if "logq_b80" in args.exp_kernel and valid <= (keys // 128 + 1) // 2 * 128:
    raise ValueError("logq_b80 requires at least one valid key in each half")

qh, kh, vh = (x.to("hpu") for x in (q, k, v))
kt = kh.T.contiguous()
bound = (qh.float().norm(dim=-1) * kh[:valid].float().norm(dim=-1).max()).bfloat16().reshape(128, 1)
output = torch.full_like(qh, -1)
row_sum = torch.full((128, 1), -1, dtype=torch.float32, device="hpu")
guard_in = torch.full((4096,), 0xCD, dtype=torch.uint8, device="hpu")
guard_out = torch.zeros_like(guard_in)
check = torch.randn(128, 128).to("hpu")
mm = torch.compile(lambda x: x @ x, backend="hpu_backend", fullgraph=True)
before = mm(check).cpu()
torch.hpu.synchronize()
info = (C.c_uint64 * 5)()
assert lib.context_info(0, info) == 0
assert lib.context_scal_nop(int(info[0])) == 0
base = info[1] + 0x200000
matrix, wide = 128 * 128 * 2, 128 * keys * 2
sq, sk, sv = base, base + matrix, base + matrix + wide
sp, so = sv + wide, sv + 2 * wide
sb, sl = so + matrix, so + matrix + 256
sg = sl + 512
assert sg + 4096 <= info[1] + info[2]
# Second buffer set for --pipelined (must match context_attention_prepare_pipelined).
sp1 = sg + 4096
so1, sl1 = sp1 + wide, sp1 + wide + matrix
assert sl1 + 512 <= info[1] + info[2], "SRAM too small for double buffering"
if args.split_pv:
    if not args.pipelined or args.exp_kernel not in ("exp_lut_split", "exp_lut_half", "exp_lut_half_m", "exp_lut_half_ms8", "exp_lut_half_logq", "exp_lut_half_logq_ms8", "exp_lut_half_logq_b80_ms8", "noop4"):
        raise ValueError("--split-pv needs --pipelined and --exp-kernel exp_lut_split")
    lib.context_attention_split_layout.argtypes = [C.c_uint64, C.c_uint, C.POINTER(C.c_uint64)]
    lib.context_attention_split_layout.restype = C.c_int
    lay = (C.c_uint64 * 17)()
    assert lib.context_attention_split_layout(base, keys, lay) == 0
    assert lay[16] <= info[1] + info[2], "SRAM too small for split layout"
    first = (keys // 128 + 1) // 2 * 128
    split_q, split_ka, split_kb, split_v = lay[0], lay[1], lay[2], lay[3]
    so, so1, sb, sl, sg = lay[6], lay[13], lay[8], lay[9], lay[10]

def copy(source, target, size):
    status = lib.context_runtime_copy(0, source, target, size)
    assert status == 0, f"Synapse copy failed: {status}"

def mme(a, b, c, cols, reduction, leading):
    status = lib.context_scal_mme_ex(a, b, c, cols, reduction, leading)
    assert status == 0, f"MME failed: {status}"

def tpc(name, addresses, cols, types):
    count = len(addresses)
    status = lib.context_scal_tpc((C.c_uint64 * count)(*addresses), (C.c_uint * count)(*cols),
                                  (C.c_uint * count)(*([128] * count)), (C.c_uint * count)(*types),
                                  count, str(root / "build" / (name + ".bin")).encode(), valid)
    assert status == 0, f"TPC {name} failed: {status}"

if args.compiler_qk:
    uploads = ((qh, split_q, matrix), (kt, split_ka, wide), (vh, split_v, wide),
               (bound, sb, 256), (guard_in, sg, 4096))
elif args.split_pv:
    # Slice on the CPU: _data_ptr of an HPU slice made contiguous pointed at the wrong data.
    kt_cpu = k.T.contiguous()
    kta, ktb = kt_cpu[:, :first].contiguous().to("hpu"), kt_cpu[:, first:].contiguous().to("hpu")
    torch.hpu.synchronize()
    uploads = ((qh, split_q, matrix), (kta, split_ka, 128 * first * 2), (ktb, split_kb, 128 * (keys - first) * 2),
               (vh, split_v, wide), (bound, sb, 256), (guard_in, sg, 4096))
else:
    uploads = ((qh, sq, matrix), (kt, sk, wide), (vh, sv, wide), (bound, sb, 256), (guard_in, sg, 4096))
for tensor, address, size in uploads:
    copy(_data_ptr(tensor), address, size)
odd_reference = reference
if args.alternate_query:
    if not args.split_pv or not args.half_max or not args.lut or not full:
        raise ValueError("--alternate-query needs --split-pv --half-max --lut and numerical validation")
    alt_address = (int(lay[16]) + 255) & ~255
    assert alt_address + matrix <= info[1] + info[2]
    alt_query = q.flip(0).contiguous().to("hpu")
    torch.hpu.synchronize()
    copy(_data_ptr(alt_query), alt_address, matrix)
    lib.context_attention_set_query_alt.argtypes = [C.c_uint64]
    lib.context_attention_set_query_alt.restype = C.c_int
    assert lib.context_attention_set_query_alt(alt_address) == 0
    odd_reference = reference.flip(0).contiguous()
if args.compiler_qk:
    back = torch.empty((128, keys), dtype=torch.bfloat16, device="hpu")
    copy(split_ka, _data_ptr(back), wide)
    assert torch.equal(back.cpu(), k.T.contiguous()), "Full compiler K^T upload mismatch"
elif args.split_pv:
    for want, address in ((kt_cpu[:, :first], split_ka), (kt_cpu[:, first:], split_kb)):
        back = torch.empty(tuple(want.shape), dtype=torch.bfloat16, device="hpu")
        copy(address, _data_ptr(back), back.numel() * 2)
        assert torch.equal(back.cpu(), want), "Split K^T upload mismatch"
samples, splits = [], []
SF_OPS = {"exp": torch.exp, "tanh": torch.tanh, "sigmoid": torch.sigmoid, "erf": torch.erf,
          "softmax": lambda t: torch.softmax(t, dim=-1), "rsqrt": torch.rsqrt, "sqrt": torch.sqrt,
          "reciprocal": torch.reciprocal, "sin": torch.sin, "cos": torch.cos, "log": torch.log,
          "gelu": torch.nn.functional.gelu, "silu": torch.nn.functional.silu}


def special_ops(x):
    # Framework special-function kernels: must be bit-identical before/after our LUT writes.
    return [f(x).cpu() for f in SF_OPS.values()]

sf_input = (torch.randn(64, 1024) * 3).to("hpu").bfloat16()
sf_input_f32 = torch.rand(64, 1024).to("hpu") + 0.5
sf_before = special_ops(sf_input) + special_ops(sf_input_f32)
if args.lut:
    lib.context_attention_set_lut.argtypes = [C.c_char_p]
    lib.context_attention_set_lut.restype = C.c_int
    lib.context_attention_set_lut_mask.argtypes = [C.c_uint]
    lib.context_attention_set_lut_mask.restype = C.c_int
    assert lib.context_attention_set_lut_mask(args.lut_mask) == 0
    status = lib.context_attention_set_lut(str(root / "build" / ("lut_gen6_logq.bin" if "logq" in args.exp_kernel else "lut_gen6.bin")).encode())
    assert status == 0, f"LUT upload failed: {status}"
if args.pipeline:
    exp_isa = str(root / "build" / (args.exp_kernel + ".bin")).encode()
    norm_isa = str(root / "build/normalize.bin").encode()
    if args.pipelined:
        lib.context_attention_prepare_pipelined.argtypes = [C.c_uint64] + [C.c_uint] * 3 + [C.c_char_p] * 2
        lib.context_attention_prepare_pipelined.restype = C.c_int
        lib.context_attention_run_pipelined.argtypes = [C.c_uint]
        lib.context_attention_run_pipelined.restype = C.c_int
        lib.context_attention_set_defer_norm.argtypes = [C.c_int]
        lib.context_attention_set_defer_norm.restype = C.c_int
        assert lib.context_attention_set_defer_norm(int(args.defer_norm)) == 0
        lib.context_attention_set_tpc_prefence.argtypes = [C.c_int]
        lib.context_attention_set_tpc_prefence.restype = C.c_int
        assert lib.context_attention_set_tpc_prefence(int(args.tpc_prefence)) == 0
        if args.split_pv:
            lib.context_attention_set_half_max.argtypes = [C.c_int]
            lib.context_attention_set_half_max.restype = C.c_int
            assert lib.context_attention_set_half_max(int(args.half_max)) == 0
            lib.context_attention_prepare_split.argtypes = [C.c_uint64] + [C.c_uint] * 3 + [C.c_char_p] * 2
            lib.context_attention_prepare_split.restype = C.c_int
            norm_name = "normalize3" if args.half_max else "normalize2"
            status = lib.context_attention_prepare_split(base, keys, valid, args.tpcs, exp_isa,
                                                          str(root / "build" / (norm_name + ".bin")).encode())
        else:
            status = lib.context_attention_prepare_pipelined(base, keys, valid, args.tpcs, exp_isa, norm_isa)
    else:
        status = lib.context_attention_prepare_mode(base, keys, valid, args.tpcs, exp_isa, norm_isa,
                                                    MODES.index(args.mode))
    assert status == 0, f"Pipeline prepare failed: {status}"
    poison = torch.full_like(output, -1)
    torch.hpu.synchronize()
    for iteration in range(args.repeats):
        copy(_data_ptr(poison), so, matrix)
        if args.pipelined:
            copy(_data_ptr(poison), so1, matrix)
        start = time.perf_counter()
        if args.pipelined:
            status = lib.context_attention_run_pipelined(args.pipelined)
        elif args.blocks:
            status = lib.context_attention_run_n(args.blocks)
        else:
            status = lib.context_attention_run()
        samples.append((time.perf_counter() - start) * 1000 / max(args.blocks, args.pipelined, 1))
        assert status == 0, f"Pipeline run {iteration} failed: {status}"
        host_us, used = (C.c_double * 3)(), C.c_uint()
        assert lib.context_attention_last_timing(host_us, C.byref(used)) == 0
        splits.append(list(host_us))
        rms = None
        if args.split_pv and iteration == 0:
            # Debug dump of parity-0 buffers (block n-2 when n is even).
            def grab(addr, shape, dtype):
                t = torch.empty(shape, dtype=dtype, device="hpu")
                copy(addr, _data_ptr(t), t.numel() * t.element_size())
                return t.cpu()
            dump = {"pa": grab(lay[4], (128, first), torch.bfloat16), "pb": grab(lay[5], (128, keys - first), torch.bfloat16),
                    "oa": grab(lay[6], (128, 128), torch.bfloat16), "ob": grab(lay[7], (128, 128), torch.bfloat16),
                    "sum": grab(lay[9], (128,), torch.float32), "q": q, "k": k, "v": v, "valid": valid, "first": first,
                    "sram_q": grab(lay[0], (128, 128), torch.bfloat16),
                    "sram_ka": grab(lay[1], (128, first), torch.bfloat16),
                    "sram_kb": grab(lay[2], (128, keys - first), torch.bfloat16),
                    "sram_v": grab(lay[3], (keys, 128), torch.bfloat16),
                    "layout": [int(x) for x in lay], "base": int(base)}
            if args.compiler_qk:
                p_all = grab(lay[4], (128, keys), torch.bfloat16)
                dump["pa"], dump["pb"] = p_all[:, :first].contiguous(), p_all[:, first:].contiguous()
                dump["sram_ka"], dump["sram_kb"] = k.T[:, :first].contiguous(), k.T[:, first:].contiguous()
            torch.save(dump, root / "evidence" / "split-debug.pt")
            sref = (q.float() @ k.float().T).bfloat16().float()
            def stat(name, x):
                print("SPLIT_DEBUG", name, "finite", bool(torch.isfinite(x.float()).all()),
                      "min", float(x.float().nan_to_num().min()), "max", float(x.float().nan_to_num().max()), flush=True)
            for name in ("pa", "pb", "oa", "ob", "sum"):
                stat(name, dump[name])
        if full:
            copy(so, _data_ptr(output), matrix)
            observed = output.cpu().float()
            rms = ((observed - reference).square().mean() / reference.square().mean().clamp_min(1e-30)).sqrt().item()
            assert torch.isfinite(observed).all() and rms < 0.01, f"Replay {iteration}: {rms}"
            if iteration == 0:
                import hashlib
                print("OUTPUT_SHA256 " + hashlib.sha256(output.cpu().view(torch.int16).numpy().tobytes()).hexdigest(), flush=True)
            if args.pipelined:
                copy(so1, _data_ptr(output), matrix)
                odd = output.cpu().float()
                rms1 = ((odd - odd_reference).square().mean() / odd_reference.square().mean().clamp_min(1e-30)).sqrt().item()
                assert torch.isfinite(odd).all() and rms1 < 0.01, f"Replay {iteration}: odd buffer {rms1}"
                if not args.alternate_query:
                    assert abs(rms1 - rms) < 1e-6, f"Replay {iteration}: parity mismatch {rms1} vs {rms}"
                else:
                    assert torch.allclose(odd.flip(0), observed, rtol=0, atol=0), "Alternating query outputs differ after unpermuting"
        assert torch.equal(mm(check).cpu(), before), "Framework GEMM changed between replays"
        print("PIPELINE_REPLAY " + json.dumps({"iteration": iteration, "mode": args.mode, "ms": samples[-1],
            "enqueue_us": host_us[0], "submit_us": host_us[1], "wait_us": host_us[2],
            "command_bytes": used.value, "rms": rms}), flush=True)
    elapsed = sum(samples) / len(samples)
else:
    start = time.perf_counter()
    first = min(keys, 32768)
    mme(sq, sk, sp, first, 128, keys)
    if keys > first:
        mme(sq, sk + first * 2, sp + first * 2, keys - first, 128, keys)
    tpc("exp_sum", [sp, sb, sp, sl], [keys, 1, keys, 1], [6, 6, 6, 7])
    mme(sp, sv, so, 128, keys, 128)
    tpc("normalize", [so, sl, so], [128, 1, 128], [6, 7, 6])
    elapsed = (time.perf_counter() - start) * 1000
copy(so, _data_ptr(output), matrix)
if args.half_max:
    stats = torch.empty((256, 2), dtype=torch.float32, device="hpu")
    copy(sl, _data_ptr(stats), 2048)
    row_sum = stats[:, 1:].contiguous()
else:
    copy(sl, _data_ptr(row_sum), 512)
copy(sg, _data_ptr(guard_out), 4096)
actual = output.cpu().float()
error = None
if full:
    error = ((actual - reference).square().mean() / reference.square().mean().clamp_min(1e-30)).sqrt().item()
    assert torch.isfinite(actual).all() and torch.isfinite(row_sum.cpu()).all()
    assert torch.all(row_sum.cpu() > 0)
    assert error < 0.01, f"FP32 reference RMS mismatch {error}"
assert torch.equal(guard_out.cpu(), guard_in.cpu()), "SRAM output guard changed"
assert torch.equal(qh.cpu(), q) and torch.equal(kh.cpu(), k) and torch.equal(vh.cpu(), v)
assert torch.equal(mm(check).cpu(), before), "Framework GEMM changed after SRAM attention"
assert torch.equal((qh + 2).cpu(), q + 2), "Framework vector operation changed"
sf_after = special_ops(sf_input) + special_ops(sf_input_f32)
sf_names = [f"{k}-{n}" for k in ("bf16", "f32") for n in SF_OPS]
sf_same = {n: torch.allclose(a, b, rtol=0, atol=0, equal_nan=True) for n, a, b in zip(sf_names, sf_before, sf_after)}
print("SPECIAL_OPS_SAME " + json.dumps(sf_same), flush=True)
if args.lut:
    torch.save({"before": sf_before, "after": sf_after, "names": sf_names}, root / "evidence" / "lut-special-ops.pt")
assert all(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True) for a, b in zip(sf_before, sf_after)), "Framework special functions changed"
print("RESULT_RUNTIME_ATTENTION " + json.dumps({"queries": 128, "keys": keys, "valid_keys": valid,
    "fp32_relative_rms": error, "guard_ok": True, "inputs_unchanged": True, "torch_after": True,
    "pipeline": args.pipeline, "mode": args.mode, "exp_kernel": args.exp_kernel, "rearm_only": args.rearm_only, "blocks": args.blocks, "pipelined": args.pipelined, "defer_norm": args.defer_norm, "tpc_prefence": args.tpc_prefence, "split_pv": args.split_pv, "tpc_rate": args.tpc_rate, "half_max": args.half_max, "dual_qk": args.dual_qk, "wreg_bulk": args.wreg_bulk, "lut": args.lut, "lut_mask": args.lut_mask, "special_ops_unchanged": True, "tpcs": args.tpcs if args.pipeline else 1,
    "compiler_qk": args.compiler_qk, "alternate_query": args.alternate_query, "host_split_us": splits,
    "cached_pipeline_replay_ms": samples, "probe_ms": elapsed}), flush=True)
