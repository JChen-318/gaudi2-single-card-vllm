# SPDX-License-Identifier: MIT
"""Distinct-query SRAM batch, retaining and returning EVERY output and row statistic.

Quiesced isolated process only. Reports HPU-HBM -> SRAM DMA, device computation,
SRAM -> HPU-HBM DMA separately. CPU references/checks and preparation excluded.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time
import torch
import habana_frameworks.torch.core  # noqa: F401
from habana_frameworks.torch.utils.experimental import _data_ptr
import mme_dump

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path)
p.add_argument("--diagnostics", type=Path, help="Save CPU stats/output and final P tiles when numerical validation fails")
p.add_argument("--keys", type=int, default=1024)
p.add_argument("--blocks", type=int, default=4)
p.add_argument("--group-blocks", type=int, default=0, help="Stream this many tiles per native batch; preserve all outputs")
p.add_argument("--parallel-pv", action="store_true", help="256 rows: assign each key half PV to one MME")
p.add_argument("--coalesce-qk", action="store_true", help="Merge equivalent QK width activations")
p.add_argument("--tile-rows", type=int, choices=(128,256,512), default=128)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--tpcs", type=int, default=24)
p.add_argument("--reload-kv", action="store_true", help="Include K/V reload in each batch, instead of reusing a head")
p.add_argument("--reload-head-kv", action="store_true", help="Include one K/V load per complete streamed head")
p.add_argument("--scores-fp16", action="store_true", help="MME stores scores in FP16, exp overwrites them with BF16 P")
p.add_argument("--exp-kernel", default="exp_lut_half_logq_b80_ms8",
               choices=("exp_f16_tile_fixed0_g15pad", "exp_f16_tile_fixed0_g16pad", "exp_f16_tile_fixed0_g8pad", "exp_f16_tile_fixed0_g8", "exp_f16_tile_fixed0_idx", "exp_f16_tile_affine_nomain_ms8", "exp_f16_tile_affine_skew_ms8", "exp_f16_tile_affine_skewidx_ms8", "exp_lut_half_logq_b80_ms8", "exp_lut_half_logq", "exp_lut_half_m", "exp_f16_logq", "exp_f16_logq_ms8", "exp_f16_tile_logq_ms8", "exp_f16_affine_ms8", "exp_f16_affine_np_ms8", "exp_f16_tile_affine_ms8", "noop_exp3"))
p.add_argument("--interleave-arm", action="store_true", help="Arm each block's monitors just before its ECBs")
p.add_argument("--ablation", choices=("none", "tpc-only", "mme-only"), default="none",
               help="tpc-only: skip all MME work; mme-only: skip exp, PV fences released at once (profile-only)")
p.add_argument("--overlap-dma", action="store_true",
               help="Two Q/O SRAM slots; DMA group g+1 input and g-1 output while group g computes")
p.add_argument("--split-fence", action="store_true",
               help="128/512 rows: S-reuse QK credits on MME fence 2, exp->PV credits on fence 3")
p.add_argument("--prefetch-desc", action="store_true",
               help="128/512 rows: write MME descriptors before the fence, commit after it")
p.add_argument("--mme-no-barrier", action="store_true",
               help="Experiment (128/512 rows): MME descriptor writes and program end without engine barrier")
p.add_argument("--cross-group", action="store_true",
               help="512 rows: keep two groups in flight (needs --overlap-dma --interleave-arm --split-fence)")
p.add_argument("--multi-monitor", action="store_true",
               help="TPC fences: one armed multi-write monitor per four TPCs (wr_num=3)")
p.add_argument("--tpc-cmd", type=lambda x: int(x, 0), default=0xf,
               help="Experiment: TPC_CMD invalidate mask for delta exp launches (must include dcache 0x2)")
p.add_argument("--local-qk-barrier",action="store_true",help="512 rows: proven even/odd row ownership, local MME barrier instead of global S-reuse monitor")
p.add_argument("--relaxed-qk-reuse",action="store_true",help="Isolated experiment: omit the same-engine QK reuse barrier")
p.add_argument("--fixed0-exponent-shift",type=int,choices=(0,8),default=0,help="Fold common P scale 2^-shift into LUT; no extra TPC instruction")
p.add_argument("--wide-query",action="store_true",help="Experimental two M512 rows per commit (M1024), K/V read from HBM")
p.add_argument("--score-row-pad",type=int,choices=(0,128,384,640),default=0,help="Experimental physical S row padding; logical scores unchanged")
p.add_argument("--key-chunk-quantum",type=int,choices=(128,512),default=128,help="512 rows: align first three key chunks to this many columns")
p.add_argument("--mme-async-commit", action="store_true", help="Isolated experiment: commit without engine barrier, retain split dependency fences")
p.add_argument("--mme-pack", type=int, choices=(0,1,2,4), default=0, help="512 rows: consecutive MME operations per static CQ batch")
p.add_argument("--profile-only", action="store_true",
               help="Timing ablation: outputs are NOT valid attention; numeric checks are skipped")
a = p.parse_args()
if (a.exp_kernel in ("noop_exp3", "exp_f16_tile_affine_nomain_ms8") or a.ablation != "none") and not a.profile_only:
    raise ValueError("noop exp / ablations produce invalid outputs: pass --profile-only")
if a.fixed0_exponent_shift and not a.exp_kernel.startswith("exp_f16_tile_fixed0_"):
    raise ValueError("Exponent shift requires fixed0 exp")
if a.relaxed_qk_reuse and not a.local_qk_barrier:
    raise ValueError("Relaxed reuse requires local row-ownership proof")
if a.local_qk_barrier and not (a.mme_async_commit and not a.prefetch_desc):
    raise ValueError("Local QK barrier requires async commit and no descriptor prefetch")
if (a.key_chunk_quantum != 128 or a.score_row_pad) and a.tile_rows != 512:
    raise ValueError("Key chunk alignment is implemented for 512 rows only")
if a.mme_async_commit and not (a.tile_rows==512 and a.interleave_arm and a.split_fence and a.mme_no_barrier):
    raise ValueError("Async commit: 512 rows, interleave, split fences, --mme-no-barrier")
if a.mme_pack and not (a.tile_rows==512 and a.interleave_arm and a.split_fence):
    raise ValueError("MME packing requires 512 rows, interleaved arming, split fences")
if a.split_fence and a.tile_rows == 256:
    raise ValueError("--split-fence is implemented for 128- and 512-row batches")
if a.ablation != "none" and a.tile_rows == 256:
    raise ValueError("--ablation: 128 or 512 rows")
if a.ablation == "tpc-only" and a.tile_rows == 512 and not a.interleave_arm:
    raise ValueError("512-row tpc-only ablation is implemented for --interleave-arm")
torch.set_num_threads(8)
torch.manual_seed(71)
root = Path(__file__).resolve().parent
keys = valid = a.keys
n = a.blocks
if a.input:
    data = bytearray(a.input.read_bytes())
    magic, keys, valid, dim, n, rows, reserved = struct.unpack_from("<8s6I", data)
    if magic != b"G2BATCH1" or dim != 128 or rows != n * 128 or not 0 <= reserved <= rows:
        raise ValueError("Invalid batch input")
    offset, tensors = 32, []
    for shape in ((n * 128, 128), (keys, 128), (keys, 128)):
        count = shape[0] * shape[1]
        tensors.append(torch.frombuffer(data, dtype=torch.bfloat16, count=count, offset=offset).clone().reshape(shape))
        offset += count * 2
    q, k, v = tensors
    reference = torch.frombuffer(data, dtype=torch.float32, count=n * 16384, offset=offset).clone().reshape(n, 128, 128)
else:
    q = (torch.randint(-4, 5, (n * 128, 128)).float() / 16).bfloat16()
    k = (torch.randint(-4, 5, (keys, 128)).float() / 16).bfloat16()
    v = (torch.randint(-4, 5, (keys, 128)).float() / 4).bfloat16()
    reference = torch.cat([torch.softmax(q[i * 128:(i + 1) * 128].float() @ k.float().T, -1) @ v.float()
                           for i in range(n)]).reshape(n, 128, 128)
g = a.group_blocks or n
if a.wide_query and not (a.tile_rows==512 and g%8==0 and a.exp_kernel.startswith("exp_f16_tile_fixed0_") and not a.score_row_pad and not a.mme_pack):
    raise ValueError("Wide-query needs fixed0 quad, group multiple of 8, no row padding/packing")
valid_queries = (reserved or rows) if a.input else n * 128
if not 2 <= n <= 320 or not 2 <= g <= (24 if a.wide_query else 20) or n % g or keys % 128 or not 0 < valid <= keys or not 1 <= a.tpcs <= 24 or not 1 <= a.repeats <= 200:
    raise ValueError("Unsupported batch dimensions/replays")
if a.tile_rows == 512 and (g % 4 or g > (24 if a.wide_query else 16) or not a.scores_fp16 or not (a.exp_kernel.startswith("exp_f16_tile_") or a.exp_kernel == "noop_exp3")):
    raise ValueError("512-row tiling needs group multiple of four <=16, FP16 scores and single-tile logq")
if a.parallel_pv and a.tile_rows != 256:
    raise ValueError("Parallel PV requires --tile-rows 256")
if a.coalesce_qk and not a.scores_fp16:
    raise ValueError("Coalesced real batch requires FP16 scores")
if a.tile_rows == 256 and (g % 2 or not a.scores_fp16 or not (a.exp_kernel.startswith("exp_f16_tile_") or a.exp_kernel == "noop_exp3")):
    raise ValueError("256-row tiling requires even group, FP16 scores and exp_f16_tile_logq_ms8")
if a.tile_rows == 128 and a.exp_kernel.startswith("exp_f16_tile_"):
    raise ValueError("Single-tile exp requires --tile-rows 256")
if a.reload_kv and a.reload_head_kv:
    raise ValueError("Choose one K/V loading policy")
bound = q.float().norm(dim=-1) * k[:valid].float().norm(dim=-1).max()
if not torch.isfinite(bound).all() or bound.max() > 80:
    raise ValueError("Stable-softmax fallback required (bound >80)")
if a.exp_kernel.startswith("exp_f16_tile_fixed0_"):
    if a.tile_rows != 512:
        raise ValueError("Fixed0 currently requires 512-query / four key chunks")
    # Keep all BF16 P normal and P@V / row sums below e^86 (<FP32 max).
    # Extra .25 covers FP16 QK/affine rounding; stricter than old <=80 gate.
    upper = float(bound.max())
    vmax = float(v[:valid].float().abs().max())
    fixed_shift=a.fixed0_exponent_shift*math.log(2.)
    fixed_log_upper = upper-fixed_shift + .25 + math.log(valid) + math.log(max(1.,vmax))
    fixed_sum_log_lower=math.log(valid)-upper-fixed_shift-.25
    if not math.isfinite(vmax) or -upper-fixed_shift-.25 <= -86 or fixed_log_upper >= 86 or fixed_sum_log_lower <= math.log(1e-30):
        raise ValueError("Fixed0 overflow/FTZ/normalizer bound failed: use sampled-max kernel")
    if a.exp_kernel in ("exp_f16_tile_fixed0_g8pad","exp_f16_tile_fixed0_g16pad","exp_f16_tile_fixed0_g15pad"):
        # Exact zero sentinel, including for the shifted LUT, not a tiny exp.
        pad=-82.5 if a.fixed0_exponent_shift else -88.0625
        bias=16256-128*a.fixed0_exponent_shift
        assert struct.unpack('<e',struct.pack('<e',pad*184.625))[0]==-bias
        if bool(torch.count_nonzero(k[valid:])) or bool(torch.count_nonzero(v[valid:])):
            raise ValueError("Padded K/V must be zero")
    print("FIXED0_BOUND " + json.dumps(dict(score=upper,vmax=vmax,pv_log_upper=fixed_log_upper)),flush=True)
if valid <= (keys // 128 + 1) // 2 * 128:
    raise ValueError("This experiment requires valid keys in both halves")
assert all(not torch.equal(q[:128], q[i * 128:(i + 1) * 128]) for i in range(1, (valid_queries + 127) // 128)), "Queries must be distinct"

lib = C.CDLL(str(root / "build/libcontext_bridge.so"))
def api(name, argtypes):
    f = getattr(lib, name)
    f.argtypes, f.restype = argtypes, C.c_int
    return f
u, u64, up = C.c_uint, C.c_uint64, C.POINTER(C.c_uint64)
info_fn = api("context_info", [u, up])
init = api("context_scal_nop", [C.c_int])
prefix = {128:"context_batch",256:"context_tile",512:"context_quad"}[a.tile_rows]
layout = api(prefix + "_layout", [u64, u, u, up])
prepare = api(prefix + "_prepare", [u64, u, u, u, u, C.c_char_p, C.c_char_p])
run = api(prefix + "_run", [])
dma = api("context_batch_dma", [u, up, up, up])
dma_enqueue = api("context_batch_dma_enqueue", [u, up, up, up])
dma_sync = api("context_batch_dma_sync", [])
api("context_set_wreg_bulk", [C.c_int])(1)
api("context_attention_set_rearm_only", [C.c_int])(1)
if a.exp_kernel != "noop_exp3" and a.scores_fp16 != a.exp_kernel.startswith("exp_f16_"):
    raise ValueError("FP16 scores and exp_f16 kernel must be selected together")
parts = 4 if a.tile_rows == 512 else 2
quantum=a.key_chunk_quantum if a.tile_rows==512 else 128
first=(keys+parts*quantum-1)//(parts*quantum)*quantum
if valid <= (parts-1)*first:
    raise ValueError("All key chunks must contain valid keys")
stat_bytes = 4096 if a.tile_rows == 512 else 2048
def load_descs(shape, fp16):
    descs = mme_dump.load_set(root / "fixtures" / shape / "mme_debug")
    if not descs:
        raise ValueError("Missing compiler fixture: " + shape)
    if fp16:
        for _, _, words in descs:
            index = (0x30 - 8) // 4
            assert words[index] >> 28 == 9 and (words[index] >> 24) & 15 == 9
            words[index] = (words[index] & 0x0FFFFFFF) | 8 << 28
    return mme_dump.coalesce_qk(descs) if a.coalesce_qk and fp16 else descs

def descriptor_args(descs):
    return (len(descs), (C.c_uint32 * (182 * len(descs)))(*[w for _, _, ws in descs for w in ws]),
            (u * len(descs))(*[e for _, e, _ in descs]))
def padded_score_rows(descs, qk):
    if a.wide_query:return mme_dump.repeat_rows_512(descs)
    if not a.score_row_pad:return descs
    # Pinned M512 captures only. Rescale row coordinates of QK Cout / PV A.
    # Column bounds/steps and active row count stay unchanged.
    out=[]
    for act,e,orig in descs:
        w=list(orig);ix=lambda off:(off-8)//4
        ten=0xf0 if qk else 0x40
        width=w[ix(ten)];stride=width+a.score_row_pad
        assert width%128==0 and w[ix(ten+4)]==512*width
        row_fields=list(range(1,5))+list(range(6,10))+list(range(11,22))
        for j in row_fields:
            val=w[ix(ten+4*j)];assert val%width==0
            w[ix(ten+4*j)]=val//width*stride
        ports=[0x22c+k*20 for k in range(4)] if qk else [0x15c+port*40+slave*20 for port in (0,4) for slave in (0,1)]
        for off in ports:
            assert w[ix(off+4)]%width==0
            assert all(w[ix(off+4*j)]==0 for j in (2,3,4))
            w[ix(off+4)]=w[ix(off+4)]//width*stride
        out.append((act,e,w))
    return out

if a.tile_rows == 512:
    assert api("context_quad_set_row_pad", [u])(a.score_row_pad) == 0
    assert api("context_quad_set_key_quantum", [u])(a.key_chunk_quantum) == 0
    setter = api("context_quad_set_mme", [u, u, C.POINTER(C.c_uint32), C.POINTER(u)])
    for h,width in enumerate((first, keys-3*first)):
        assert setter(h,*descriptor_args(padded_score_rows(load_descs(f"g-m512n{width}k128",True),True))) == 0
        assert setter(h+2,*descriptor_args(padded_score_rows(load_descs(f"g-m512n128k{width}",False),False))) == 0
    kt = torch.cat([k[h*first:min((h+1)*first,keys)].T.contiguous().flatten() for h in range(4)])
elif a.tile_rows == 256:
    assert api("context_tile_set_parallel_pv", [C.c_int])(int(a.parallel_pv)) == 0
    setter = api("context_tile_set_mme", [u, u, C.POINTER(C.c_uint32), C.POINTER(u)])
    for h, width in enumerate((first, keys-first)):
        assert setter(h, *descriptor_args(load_descs(f"g-m256n{width}k128", True))) == 0
        assert setter(h+2, *descriptor_args(load_descs(f"g-m256n128k{width}", False))) == 0
    kt = torch.cat([k[:first].T.contiguous().flatten(), k[first:].T.contiguous().flatten()])
else:
    set_qk = api("context_attention_set_compiler_qk", [u, C.POINTER(C.c_uint32), C.POINTER(u)])
    assert set_qk(*descriptor_args(load_descs(f"g-m128n{keys}k128", a.scores_fp16))) == 0
    kt = k.T.contiguous()
qh, kth, vh = (t.to("hpu") for t in (q, kt, v))
out = torch.full((n, 128, 128), -7, dtype=torch.bfloat16, device="hpu")
stats = torch.empty((n, stat_bytes//8, 2), dtype=torch.float32, device="hpu")
poison = torch.full_like(out, -7)
guard = torch.full((4096,), 0xCD, dtype=torch.uint8, device="hpu")
guard_back = torch.empty_like(guard)
check = torch.randn(128, 128).to("hpu")
before = (check @ check).cpu()
ops = [torch.exp, torch.tanh, torch.sigmoid, torch.erf, lambda t: torch.softmax(t, -1), torch.rsqrt,
       torch.sqrt, torch.reciprocal, torch.sin, torch.cos, torch.log, torch.nn.functional.gelu, torch.nn.functional.silu]
sf = [(torch.randn(64, 1024) * 3).to("hpu").bfloat16(), torch.rand(64, 1024).to("hpu") + 0.5]
sf_before = [f(t).cpu() for t in sf for f in ops]
torch.hpu.synchronize()
info = (u64 * 5)()
assert info_fn(0, info) == 0 and init(int(info[0])) == 0
base = info[1] + 0x200000
if a.wide_query:
    assert api("context_quad_set_wide_kv", [u64,u64])(_data_ptr(kth),_data_ptr(vh)) == 0
L = (u64 * 10)()
assert layout(base, keys, g, L) == 0 and L[9] <= info[1] + info[2]
# Explicit kernel/table contract: a new name must select its table deliberately.
lut_name = {
    "exp_lut_half_m": "lut_gen6.bin",
    "exp_lut_half_logq_b80_ms8": "lut_gen6_logq.bin",
    "exp_lut_half_logq": "lut_gen6_logq.bin",
    "exp_f16_logq": "lut_gen6_logq.bin",
    "exp_f16_logq_ms8": "lut_gen6_logq.bin",
    "exp_f16_tile_logq_ms8": "lut_gen6_logq.bin",
    "exp_f16_affine_ms8": "lut_gen6_logq2.bin",
    "exp_f16_affine_np_ms8": "lut_gen6_logq2.bin",
    "exp_f16_tile_affine_ms8": "lut_gen6_logq2.bin",
    "noop_exp3": "lut_gen6_logq.bin",
    "exp_f16_tile_affine_nomain_ms8": "lut_gen6_logq2.bin",
    "exp_f16_tile_affine_skew_ms8": "lut_gen6_logq2.bin",
    "exp_f16_tile_affine_skewidx_ms8": "lut_gen6_logq2.bin",
    "exp_f16_tile_fixed0_idx": "lut_gen6_logq2.bin",
    "exp_f16_tile_fixed0_g8": "lut_gen6_logq2.bin",
    "exp_f16_tile_fixed0_g8pad": "lut_gen6_logq2.bin",
    "exp_f16_tile_fixed0_g16pad": "lut_gen6_logq2.bin",
    "exp_f16_tile_fixed0_g15pad": "lut_gen6_logq2.bin",
}[a.exp_kernel]
if a.tile_rows == 128:
    assert api("context_batch_set_options", [C.c_int, C.c_int])(int(a.interleave_arm), {"none":0,"tpc-only":1,"mme-only":2}[a.ablation]) == 0
elif a.tile_rows == 256:
    assert api("context_tile_set_options", [C.c_int])(int(a.interleave_arm)) == 0
else:
    assert api("context_quad_set_options", [C.c_int])(int(a.interleave_arm)) == 0
    assert api("context_quad_set_ablation", [C.c_int])({"none":0,"tpc-only":1,"mme-only":2}[a.ablation]) == 0
if a.mme_no_barrier:
    if a.tile_rows == 256:
        raise ValueError("--mme-no-barrier: 128 or 512 rows")
    assert api("context_set_mme_nobarrier_allowed", [C.c_int])(1) == 0
    assert api({128:"context_batch_set_mme_nobarrier",512:"context_quad_set_mme_nobarrier"}[a.tile_rows], [C.c_int])(1) == 0
if a.mme_async_commit:
    assert api("context_set_mme_async_commit", [C.c_int])(1) == 0
if a.tpc_cmd != 0xf:
    assert api("context_set_tpc_delta_cmd", [C.c_uint])(a.tpc_cmd) == 0
if a.multi_monitor:
    if a.tile_rows == 256:
        raise ValueError("--multi-monitor: 128 or 512 rows")
    assert api("context_set_tpc_multi_monitor", [C.c_int])(1) == 0
if a.cross_group:
    if a.tile_rows != 512 or not (a.overlap_dma and a.interleave_arm and a.split_fence):
        raise ValueError("--cross-group: 512 rows with --overlap-dma --interleave-arm --split-fence")
if a.prefetch_desc:
    if a.tile_rows == 256:
        raise ValueError("--prefetch-desc: 128 or 512 rows")
    assert api({128:"context_batch_set_prefetch_desc",512:"context_quad_set_prefetch_desc"}[a.tile_rows], [C.c_int])(1) == 0
if a.split_fence:
    assert api({128:"context_batch_set_split_fence",512:"context_quad_set_split_fence"}[a.tile_rows], [C.c_int])(1) == 0
if a.cross_group:
    assert api("context_quad_set_cross_group", [C.c_int])(1) == 0
if a.fixed0_exponent_shift:
    lut_name="lut_gen6_logq2_shift8.bin"
if a.exp_kernel in ("exp_f16_tile_fixed0_g8pad","exp_f16_tile_fixed0_g16pad","exp_f16_tile_fixed0_g15pad"):
    assert api("context_quad_set_exp_shift", [u])(a.fixed0_exponent_shift) == 0
    assert api("context_quad_set_exp_pad", [C.c_int])(1) == 0
if a.local_qk_barrier:
    assert api("context_quad_set_local_qk_barrier", [C.c_int])(1) == 0
if a.relaxed_qk_reuse:
    assert api("context_quad_set_relaxed_qk_reuse", [C.c_int])(1) == 0
if a.mme_pack:
    assert api("context_quad_set_mme_pack", [u])(a.mme_pack) == 0
assert api("context_attention_set_lut", [C.c_char_p])(str(root / "build" / lut_name).encode()) == 0
norm_name = "normalize_quad_fixed1024.bin" if a.wide_query else "normalize_quad_fixed.bin" if a.exp_kernel.startswith("exp_f16_tile_fixed0_") else {128:"normalize_batch.bin",256:"normalize_tile256.bin",512:"normalize_quad.bin"}[a.tile_rows]
assert prepare(base, keys, valid, g, a.tpcs, str(root / "build" / (a.exp_kernel + ".bin")).encode(),
               str(root / "build" / norm_name).encode()) == 0
groups = n // g
if a.overlap_dma:
    A = (u64 * 5)()
    assert api(prefix + "_layout_alt", [u64, u, u, up])(base, keys, g, A) == 0 and A[4] <= info[1] + info[2]
    prepare_slot = api(prefix + "_prepare_slot", [u, u64, u, u, u, u, C.c_char_p, C.c_char_p])
    assert prepare_slot(1, base, keys, valid, g, a.tpcs, str(root / "build" / (a.exp_kernel + ".bin")).encode(),
                        str(root / "build" / norm_name).encode()) == 0
    submit = api(prefix + "_submit", [u])
    wait = api(prefix + "_wait", [])
    # slot -> (q, OA, OB, stats)
    slot_addr = [(L[0], L[5], L[6], L[7]), (A[0], A[1], A[2], A[3])]
else:
    slot_addr = [(L[0], L[5], L[6], L[7])]

def transfer(entries):
    count = len(entries)
    return (count, (u64 * count)(*[e[0] for e in entries]), (u64 * count)(*[e[1] for e in entries]),
            (u64 * count)(*[e[2] for e in entries]))

kv_copy = [] if a.wide_query else [(_data_ptr(kth), L[1], k.numel() * 2), (_data_ptr(vh), L[2], v.numel() * 2)]
transfers = []
for group in range(n // g):
    # Explicit offsets in owned contiguous allocations, without HPU slice data_ptr.
    sq, so, _, ss = slot_addr[group % len(slot_addr)]
    q_copy = (_data_ptr(qh) + group * g * 32768, sq, g * 32768)
    inp = transfer([q_copy] + (kv_copy if a.reload_kv else []))
    ret = transfer([(so, _data_ptr(out) + group * g * 32768, g * 32768),
                    (ss, _data_ptr(stats) + group * g * stat_bytes, g * stat_bytes)])
    transfers.append((inp, ret))
fill = transfer([(_data_ptr(poison),sa[1]+h*g*32768,g*32768) for sa in slot_addr for h in range(4)] if a.tile_rows == 512 else
                [(_data_ptr(poison), x, g * 32768) for sa in slot_addr for x in (sa[1], sa[2])])
assert dma(*transfer([(_data_ptr(guard), L[8], 4096)])) == 0
t0 = time.perf_counter()
if kv_copy: assert dma(*transfer(kv_copy)) == 0
initial_kv_ms = (time.perf_counter() - t0) * 1000
samples, errors = [], None
for it in range(a.repeats):
    assert dma(*fill) == 0
    out.fill_(-7)
    torch.hpu.synchronize()
    t0 = time.perf_counter()
    input_us = compute_us = output_us = 0
    if a.reload_head_kv:
        t = time.perf_counter()
        if kv_copy: assert dma(*transfer(kv_copy)) == 0
        input_us += (time.perf_counter() - t) * 1e6
    if a.cross_group:
        # Two groups in flight: group g+1 is already queued when group g ends.
        submit2 = api("context_quad_submit2", [u, C.c_int])
        t = time.perf_counter()
        assert dma(*transfers[0][0]) == 0
        t1 = time.perf_counter()
        assert submit2(0, 0) == 0
        if groups > 1:
            assert dma(*transfers[1][0]) == 0
            assert submit2(1, 1) == 0
        for group in range(groups):
            status = wait()
            assert status == 0, f"Batch run failed {status} at replay {it}, group {group}"
            assert dma_enqueue(*transfers[group][1]) == 0
            if group + 2 < groups:
                assert dma_enqueue(*transfers[group + 2][0]) == 0
            assert dma_sync() == 0
            if group + 2 < groups:
                assert submit2((group + 2) & 1, 1) == 0, f"submit failed at replay {it}, group {group + 2}"
        t2 = time.perf_counter()
        input_us += (t1-t)*1e6; compute_us += (t2-t1)*1e6
    elif a.overlap_dma:
        # Group g computes in slot g&1 while the copy stream brings group g+1's Q
        # into the other slot and returns group g-1's outputs from that slot.
        t = time.perf_counter()
        assert dma(*transfers[0][0]) == 0
        t1 = time.perf_counter()
        for group in range(groups):
            assert submit(group & 1) == 0, f"submit failed at replay {it}, group {group}"
            if group + 1 < groups:
                assert dma_enqueue(*transfers[group + 1][0]) == 0
            if group:
                assert dma_enqueue(*transfers[group - 1][1]) == 0
            assert dma_sync() == 0
            status = wait()
            assert status == 0, f"Batch run failed {status} at replay {it}, group {group}"
        t2 = time.perf_counter()
        assert dma(*transfers[-1][1]) == 0
        t3 = time.perf_counter()
        input_us += (t1-t)*1e6; compute_us += (t2-t1)*1e6; output_us += (t3-t2)*1e6
    for group, (inp, ret) in enumerate([] if a.overlap_dma else transfers):
        t = time.perf_counter()
        assert dma(*inp) == 0
        t1 = time.perf_counter()
        status = run()
        t2 = time.perf_counter()
        assert status == 0, f"Batch run failed {status} at replay {it}, group {group}"
        assert dma(*ret) == 0
        t3 = time.perf_counter()
        input_us += (t1-t)*1e6
        compute_us += (t2-t1)*1e6
        output_us += (t3-t2)*1e6
    samples.append(dict(input_us=input_us, compute_us=compute_us, output_us=output_us,
                        total_us=(time.perf_counter()-t0)*1e6))
    # Outside timing: verify EVERY tile on EVERY replay, not just parity 0/1.
    got, st = out.cpu().float(), stats.cpu()
    if a.profile_only:
        assert torch.equal((check @ check).cpu(), before), "Framework GEMM changed"
        print("BATCH_REPLAY_PROFILE_ONLY " + json.dumps(dict(iteration=it, **samples[-1])), flush=True)
        continue
    errors = ((got-reference).square().mean((1,2)) / reference.square().mean((1,2)).clamp_min(1e-30)).sqrt()
    assert torch.isfinite(got).all() and torch.isfinite(st).all() and torch.all(st[:, :, 1] > 0)
    if a.diagnostics and errors.max() >= 0.01:
        debug = {"output": got, "stats": st, "tile_rows": a.tile_rows}
        if a.tile_rows == 256:
            for h, width in enumerate((first, keys-first)):
                pp = torch.empty((256,width),dtype=torch.bfloat16,device="hpu")
                assert dma(*transfer([(L[3+h],_data_ptr(pp),pp.numel()*2)])) == 0
                debug["prob"+str(h)] = pp.cpu()
        torch.save(debug,a.diagnostics)
    assert errors.max() < 0.01, f"Per-block RMS: {errors.tolist()}"
    assert torch.equal((check @ check).cpu(), before), "Framework GEMM changed"
    print("BATCH_REPLAY " + json.dumps(dict(iteration=it, **samples[-1], max_rms=errors.max().item())), flush=True)
assert dma(*transfer([(L[8], _data_ptr(guard_back), 4096)])) == 0
assert torch.equal(guard.cpu(), guard_back.cpu())
last_q = slot_addr[(groups - 1) % len(slot_addr)][0]
for tensor, address in ((qh, last_q), (kth, L[1]), (vh, L[2])):
    expected = tensor.cpu()[-g * 128:] if address == last_q else tensor.cpu()
    back = torch.empty(expected.shape, dtype=tensor.dtype, device="hpu")
    assert dma(*transfer([(address, _data_ptr(back), back.numel()*2)])) == 0
    assert torch.equal(back.cpu(), expected), "SRAM input changed"
sf_after = [f(t).cpu() for t in sf for f in ops]
assert all(torch.allclose(x,y,rtol=0,atol=0,equal_nan=True) for x,y in zip(sf_before,sf_after)), "Framework special ops changed"
steady = samples[1:] or samples
if a.profile_only:
    print("RESULT_PROFILE_ONLY " + json.dumps(dict(valid_attention=False, blocks=n, group_blocks=g, groups=n//g,
        tile_rows=a.tile_rows, exp_kernel=a.exp_kernel, ablation=a.ablation, interleave_arm=a.interleave_arm,
        overlap_dma=a.overlap_dma, split_fence=a.split_fence, prefetch_desc=a.prefetch_desc, mme_no_barrier=a.mme_no_barrier, cross_group=a.cross_group, multi_monitor=a.multi_monitor, tpc_cmd=a.tpc_cmd, mme_pack=a.mme_pack, mme_async_commit=a.mme_async_commit, key_chunk_quantum=a.key_chunk_quantum, local_qk_barrier=a.local_qk_barrier, score_row_pad=a.score_row_pad, wide_query=a.wide_query, relaxed_qk_reuse=a.relaxed_qk_reuse, fixed0_exponent_shift=a.fixed0_exponent_shift,
        coalesce_qk=a.coalesce_qk, reload_head_kv=a.reload_head_kv, samples=samples,
        median_us={k:statistics.median(s[k] for s in steady) for k in samples[0]},
        median_compute_us_per_block=statistics.median(s['compute_us'] for s in steady)/n,
        guard_ok=True, sram_inputs_unchanged=True, special_ops_unchanged=True)), flush=True)
    raise SystemExit(0)
print("RESULT_REAL_BATCH " + json.dumps(dict(blocks=n, queries=n*128, valid_queries=valid_queries,
    group_blocks=g, groups=n//g, keys=keys, valid_keys=valid, tile_rows=a.tile_rows,
    all_outputs_checked=True, all_stats_returned=True, distinct_queries=True, reload_kv=a.reload_kv,
    reload_head_kv=a.reload_head_kv,
    exp_kernel=a.exp_kernel, initial_kv_ms=initial_kv_ms, samples=samples,
    scores_fp16=a.scores_fp16, coalesce_qk=a.coalesce_qk, parallel_pv=a.parallel_pv, interleave_arm=a.interleave_arm,
    overlap_dma=a.overlap_dma, split_fence=a.split_fence, prefetch_desc=a.prefetch_desc, mme_no_barrier=a.mme_no_barrier, cross_group=a.cross_group, multi_monitor=a.multi_monitor, tpc_cmd=a.tpc_cmd, mme_pack=a.mme_pack, mme_async_commit=a.mme_async_commit, key_chunk_quantum=a.key_chunk_quantum, local_qk_barrier=a.local_qk_barrier, score_row_pad=a.score_row_pad, wide_query=a.wide_query, relaxed_qk_reuse=a.relaxed_qk_reuse, fixed0_exponent_shift=a.fixed0_exponent_shift,
    median_us={k:statistics.median(s[k] for s in steady) for k in samples[0]},
    median_compute_us_per_block=statistics.median(s['compute_us'] for s in steady)/n,
    median_total_us_per_block=statistics.median(s['total_us'] for s in steady)/n,
    block_rms=errors.tolist(), max_rms=errors.max().item(), guard_ok=True, sram_inputs_unchanged=True,
    torch_after=True, special_ops_unchanged=True, valid_output_sha256=hashlib.sha256(out.cpu().reshape(-1,128)[:valid_queries].contiguous().view(torch.int16).numpy().tobytes()).hexdigest(), output_sha256=hashlib.sha256(out.cpu().view(torch.int16).numpy().tobytes()).hexdigest())), flush=True)
