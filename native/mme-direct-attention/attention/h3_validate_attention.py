# SPDX-License-Identifier: MIT
"""Direct MME/TPC attention for MiniMax H3 max-free layers (experimental, opt-in).

Replaces vllm_gaudi.omni.attention.max_free_attention for the one validated
shape: query/key/value [1, H, 37750, 128] BF16, unmasked, non-causal. The plan
is the probe's validated 512-row/1024-query ("wide") configuration, prepared
once per process on fixed staging tensors, because SRAM/HBM addresses are baked
into the static engine programs:

  per layer : scale Q on device; build all heads' K^T key-quarter layout on
              device; per-head numeric gates (same as batch_attention_probe.py
              fixed0 gates) computed on device, read back once.
  staging   : copy all heads into persistent Q/K^T/V tensors, synchronize once.
  per head  : select the validated address bank; run 19 cross-group submissions
              (Q in / O out by Synapse DMA) into a fresh per-call result tensor.
  fallback  : any head failing a gate, and any other shape, uses the original
              implementation (passed in).

Requirements: SCAL_CONFIG_FILE_PATH must point to scal-attention-h3-1.24.1.json
(private SOB/monitor pools) before the device is opened; the process must not
run other device work concurrently with a call (bulk staging synchronizes first).
"""
from __future__ import annotations

import ctypes as C
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("H3DIRECT_ROOT", "/home/a/gsv-data/media/mme-direct-20261108/repro/attention"))
KEYS, VALID, DIM = 37760, 37750, 128
GROUP = 16                      # 128-query blocks per native group
BLOCKS = 304                    # 37750 queries padded to 19 groups of 16 blocks
PARTS, QUANTUM = 4, 128
FIRST = (KEYS + PARTS * QUANTUM - 1) // (PARTS * QUANTUM) * QUANTUM   # 9472
STAT_BYTES = 4096
POW2_Q = os.environ.get("H3DIRECT_POW2Q", "0") == "1"
INPUT_Q_SCALE = .0625 if POW2_Q else None
# Bound the actual exp input, including the half coefficient's temperature.
SCORE_FACTOR = 261.25 / 184.6649652337873 if POW2_Q else 1.0
EXP_KERNEL = "exp_f16_tile_fixed0_pow2q_g15pad" if POW2_Q else "exp_f16_tile_fixed0_g15pad"
EXP_SHIFT = 8
LUT = "lut_gen6_logq2_shift8.bin"
NORM = "normalize_quad_fixed1024.bin"


class _Plan:
    def __init__(self, relaxed: bool, heads: int):
        self.heads=heads
        # Borrowed raw QMAN programs assume the runtime has initialized each
        # engine's addressing context. QKV dump tests may have used only DMA/
        # TPC so far, unlike the real model's QKV projections and old probes.
        warm=torch.ones((128,128),dtype=torch.bfloat16,device="hpu")
        got=(warm@warm).cpu()
        if not bool(torch.all(got==128)):
            raise RuntimeError("framework MME initialization check failed")
        torch.hpu.synchronize()
        from habana_frameworks.torch.utils.experimental import _data_ptr
        sys.path.insert(0, str(ROOT))
        import mme_dump  # noqa: E402
        self._data_ptr = _data_ptr
        lib = C.CDLL(str(ROOT / "build/libcontext_bridge.so"))
        self.lib = lib

        def api(name, argtypes):
            f = getattr(lib, name)
            f.argtypes, f.restype = argtypes, C.c_int
            return f
        u, u64, up = C.c_uint, C.c_uint64, C.POINTER(C.c_uint64)
        self.u64 = u64
        api("context_set_wreg_bulk", [C.c_int])(1)
        api("context_attention_set_rearm_only", [C.c_int])(1)
        assert api("context_quad_set_row_pad", [u])(0) == 0
        assert api("context_quad_set_key_quantum", [u])(QUANTUM) == 0

        def load(shape, fp16):
            descs = mme_dump.load_set(ROOT / "fixtures" / shape / "mme_debug")
            if not descs:
                raise RuntimeError("missing compiler fixture " + shape)
            if fp16:
                for _, _, words in descs:
                    i = (0x30 - 8) // 4
                    assert words[i] >> 28 == 9 and (words[i] >> 24) & 15 == 9
                    words[i] = (words[i] & 0x0FFFFFFF) | 8 << 28
                descs = mme_dump.coalesce_qk(descs)
            return mme_dump.repeat_rows_512(descs)

        def dargs(descs):
            return (len(descs), (C.c_uint32 * (182 * len(descs)))(*[w for _, _, ws in descs for w in ws]),
                    (u * len(descs))(*[e for _, e, _ in descs]))
        setter = api("context_quad_set_mme", [u, u, C.POINTER(C.c_uint32), C.POINTER(u)])
        for h, width in enumerate((FIRST, KEYS - 3 * FIRST)):
            assert setter(h, *dargs(load(f"g-m512n{width}k128", True))) == 0
            assert setter(h + 2, *dargs(load(f"g-m512n128k{width}", False))) == 0

        dev = "hpu"
        self.qh = torch.zeros((heads, BLOCKS * 128, DIM), dtype=torch.bfloat16, device=dev)
        self.kth = torch.zeros((heads, DIM * KEYS), dtype=torch.bfloat16, device=dev)
        self.vh = torch.zeros((heads, KEYS, DIM), dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros((BLOCKS, 128, DIM), dtype=torch.bfloat16, device=dev)
        self.stats = torch.zeros((BLOCKS, STAT_BYTES // 8, 2), dtype=torch.float32, device=dev)
        torch.hpu.synchronize()

        info = (u64 * 5)()
        assert api("context_info", [u, up])(0, info) == 0
        assert api("context_scal_nop", [C.c_int])(int(info[0])) == 0
        base = info[1] + 0x200000
        dp = _data_ptr
        assert api("context_quad_set_wide_kv", [u64, u64])(dp(self.kth), dp(self.vh)) == 0
        L = (u64 * 10)()
        assert api("context_quad_layout", [u64, u, u, up])(base, KEYS, GROUP, L) == 0 and L[9] <= info[1] + info[2]
        assert api("context_quad_set_options", [C.c_int])(1) == 0
        assert api("context_quad_set_ablation", [C.c_int])(0) == 0
        assert api("context_set_mme_nobarrier_allowed", [C.c_int])(1) == 0
        assert api("context_quad_set_mme_nobarrier", [C.c_int])(1) == 0
        assert api("context_set_mme_async_commit", [C.c_int])(1) == 0
        assert api("context_set_tpc_delta_cmd", [C.c_uint])(0xa) == 0
        assert api("context_set_tpc_multi_monitor", [C.c_int])(1) == 0
        assert api("context_quad_set_split_fence", [C.c_int])(1) == 0
        assert api("context_quad_set_cross_group", [C.c_int])(1) == 0
        assert api("context_quad_set_exp_shift", [u])(EXP_SHIFT) == 0
        if POW2_Q:
            assert api("context_quad_set_exp_pow2_q", [u])(EXP_SHIFT) == 0
        assert api("context_quad_set_exp_pad", [C.c_int])(1) == 0
        assert api("context_quad_set_local_qk_barrier", [C.c_int])(1) == 0
        if relaxed:
            assert api("context_quad_set_relaxed_qk_reuse", [C.c_int])(1) == 0
        assert api("context_attention_set_lut", [C.c_char_p])(str(ROOT / "build" / LUT).encode()) == 0
        exp_bin = str(ROOT / "build" / (EXP_KERNEL + ".bin")).encode()
        norm_bin = str(ROOT / "build" / NORM).encode()
        assert api("context_quad_prepare", [u64, u, u, u, u, C.c_char_p, C.c_char_p])(
            base, KEYS, VALID, GROUP, 24, exp_bin, norm_bin) == 0
        A = (u64 * 5)()
        assert api("context_quad_layout_alt", [u64, u, u, up])(base, KEYS, GROUP, A) == 0 and A[4] <= info[1] + info[2]
        assert api("context_quad_prepare_slot", [u, u64, u, u, u, u, C.c_char_p, C.c_char_p])(
            1, base, KEYS, VALID, GROUP, 24, exp_bin, norm_bin) == 0
        bank_status=api("context_quad_prepare_head_banks", [u])(heads)
        if bank_status:raise RuntimeError(f"head bank preparation failed: {bank_status}")
        self.select_head=api("context_quad_select_head", [u])
        self.submit2 = api("context_quad_submit2", [u, C.c_int])
        self.wait = api("context_quad_wait", [])
        self.dma = api("context_batch_dma", [u, up, up, up])
        self.dma_enqueue = api("context_batch_dma_enqueue", [u, up, up, up])
        self.dma_sync = api("context_batch_dma_sync", [])
        slots = [(L[0], L[5], L[7]), (A[0], A[1], A[3])]

        def transfer(entries):
            n = len(entries)
            return (n, (u64 * n)(*[e[0] for e in entries]), (u64 * n)(*[e[1] for e in entries]),
                    (u64 * n)(*[e[2] for e in entries]))
        self.head_bytes=BLOCKS*128*DIM*2
        self.transfers = [[] for _ in range(heads)]
        for head in range(heads):
            for group in range(BLOCKS // GROUP):
                sq, so, ss = slots[group % 2]
                inp = transfer([(dp(self.qh)+head*self.head_bytes+group*GROUP*32768, sq, GROUP*32768)])
                ret = transfer([(so, dp(self.out)+group*GROUP*32768, GROUP*32768),
                                (ss, dp(self.stats)+group*GROUP*STAT_BYTES, GROUP*STAT_BYTES)])
                self.transfers[head].append((inp,ret))

    def run_head(self, head: int, output_address: int):
        """All framework work must be quiesced. Output points into the caller's
        fresh padded tensor; head plans only bind K/V in persistent buffers."""
        assert self.select_head(head)==0
        transfers=self.transfers[head]
        for group,(_,ret) in enumerate(transfers):
            ret[2][0]=output_address+group*GROUP*32768
        groups = len(transfers)
        assert self.dma(*transfers[0][0]) == 0
        assert self.submit2(0, 0) == 0
        assert self.dma(*transfers[1][0]) == 0
        assert self.submit2(1, 1) == 0
        for g in range(groups):
            status = self.wait()
            if status:
                raise RuntimeError(f"direct attention group {g} failed: {status}")
            assert self.dma_enqueue(*transfers[g][1]) == 0
            if g + 2 < groups:
                assert self.dma_enqueue(*transfers[g + 2][0]) == 0
            assert self.dma_sync() == 0
            if g + 2 < groups:
                assert self.submit2((g + 2) & 1, 1) == 0


_PLAN: _Plan | None = None
LAST_BOUNDS=[]
STATS = {"calls": 0, "direct_heads": 0, "fallback_heads": 0, "direct_s": 0.0,
         "gates_s":0.0,"pack_enqueue_s":0.0,"head_staging_wait_s":0.0,"head_compute_s":0.0}


def _plan(heads: int = 1) -> _Plan:
    global _PLAN
    if _PLAN is None:
        _PLAN = _Plan(relaxed=os.environ.get("H3DIRECT_RELAXED", "0") == "1",heads=heads)
    return _PLAN


def head_gates(qs: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_factor: float = 1.0) -> list[bool]:
    """Fixed-0 gates of batch_attention_probe.py per head. qs/key/value [H, L, D] (qs already scaled)."""
    kmax = key.float().norm(dim=-1).amax(-1)                      # [H]
    upper = (qs.float().norm(dim=-1).amax(-1) * kmax * score_factor)              # [H] = max_i |q_i| * max_j |k_j|
    vmax = value.abs().amax(dim=(-1, -2)).float()                 # [H]
    upper, vmax = upper.cpu().tolist(), vmax.cpu().tolist()
    global LAST_BOUNDS
    LAST_BOUNDS=upper
    shift = EXP_SHIFT * math.log(2.0)
    ok = []
    for b, vm in zip(upper, vmax):
        good = math.isfinite(b) and math.isfinite(vm) and b <= 80
        good = good and -b - shift - .25 > -86
        good = good and b - shift + .25 + math.log(VALID) + math.log(max(1., vm)) < 86
        good = good and math.log(VALID) - b - shift - .25 > math.log(1e-30)
        ok.append(good)
    return ok


def _direct_max_free_attention_impl(query, key, value, scale, original):
    """Drop-in for max_free_attention(query, key, value, scale) -> [B, H, L, D]."""
    B, H, Lq, D = query.shape
    if (B != 1 or D != DIM or Lq != VALID or key.shape != query.shape or value.shape != query.shape
            or query.dtype != torch.bfloat16 or query.device.type != "hpu"
            or key.dtype!=query.dtype or value.dtype!=query.dtype
            or key.device!=query.device or value.device!=query.device or not 1<=H<=64
            or query.requires_grad or key.requires_grad or value.requires_grad):
        return original(query, key, value, scale)
    if _PLAN is not None and _PLAN.heads!=H:
        return original(query,key,value,scale)
    if POW2_Q and not math.isclose(float(scale), DIM**-.5, rel_tol=0, abs_tol=1e-12):
        return original(query,key,value,scale)
    plan = _plan(H)
    t0 = time.perf_counter();before=dict(STATS)
    qs = query[0] * (INPUT_Q_SCALE if POW2_Q else scale)                                          # [H, L, D] bf16, as exported
    k, v = key[0], value[0]
    gates = head_gates(qs, k, v, SCORE_FACTOR)
    t_pack=time.perf_counter();STATS["gates_s"]+=t_pack-t0
    # Stage all heads at once. Static head banks bind K/V into these owned,
    # persistent tensors; Q/O DMA runs without intervening framework work.
    plan.qh[:,:Lq].copy_(qs)
    plan.vh[:,:Lq].copy_(v)
    # Transpose straight into each persistent K quarter. Padding stays zero;
    # avoid a full padded K allocation, four temporaries and the concat copy.
    for part in range(PARTS):
        start=part*FIRST;width=min(FIRST,KEYS-start);valid_width=min(width,VALID-start)
        target=plan.kth[:,start*DIM:(start+width)*DIM].view(H,DIM,width)
        target[:,:,:valid_width].copy_(k[:,start:start+valid_width].transpose(1,2))
    padded=torch.empty((H,BLOCKS*128,DIM),dtype=query.dtype,device=query.device)
    result=padded[:,:Lq]
    STATS["pack_enqueue_s"]+=time.perf_counter()-t_pack
    t_stage=time.perf_counter();torch.hpu.synchronize()
    STATS["head_staging_wait_s"]+=time.perf_counter()-t_stage
    output_base=plan._data_ptr(padded)
    slow = [h for h in range(H) if not gates[h]]
    for h in range(H):
        if not gates[h]:continue
        t_compute=time.perf_counter()
        plan.run_head(h,output_base+h*plan.head_bytes)
        STATS["head_compute_s"]+=time.perf_counter()-t_compute
    if slow:
        idx = torch.tensor(slow, device=query.device)
        result[idx] = original(query[:, idx], key[:, idx], value[:, idx], scale)[0]
    torch.hpu.synchronize()
    STATS["calls"] += 1
    STATS["direct_heads"] += H - len(slow)
    STATS["fallback_heads"] += len(slow)
    elapsed=time.perf_counter()-t0
    STATS["direct_s"] += elapsed
    if os.environ.get("H3DIRECT_LOG")=="1":
        print("[h3direct] "+json.dumps(dict(call=STATS["calls"],heads=H,direct_heads=H-len(slow),
              fallback_heads=slow,pow2_q=POW2_Q,
              phase_ms={k:round((STATS[k]-before[k])*1000,3) for k in before if k.endswith("_s")},wall_s=round(elapsed,6))),flush=True)
    _validate_reference(query,key,value,result,scale,slow)
    return result.unsqueeze(0)


def _validate_reference(query,key,value,result,scale,slow):
    # Optional validation only. CPU reference work is outside attention phase
    # timers, but is included in the video request; do not use it as a speed run.
    folder=os.environ.get("H3PROF_DIR")
    if not folder:return
    control=Path(folder)/"validate-control"
    rows_n=int(control.read_text().strip()) if control.exists() else 0
    if not rows_n:return
    if not 1<=rows_n<=512:raise ValueError("Invalid reference row count")
    started=time.perf_counter();h=query.shape[1]
    ranked=sorted(range(h),key=lambda i:LAST_BOUNDS[i] if math.isfinite(LAST_BOUNDS[i]) else float("inf"),reverse=True)[:6]
    heads=sorted(set(ranked+[i for i in (0,27,36,55) if i<h]))
    hi=torch.tensor(heads,device=query.device)
    ri=torch.linspace(0,VALID-1,rows_n).long().to(query.device)
    qf=query[0].index_select(0,hi).index_select(1,ri).detach().cpu().float()
    kf=key[0].index_select(0,hi).detach().cpu().float()
    vf=value[0].index_select(0,hi).detach().cpu().float()
    of=result.index_select(0,hi).index_select(1,ri).detach().cpu().float()
    ref=torch.softmax(torch.bmm(qf,kf.transpose(1,2))*scale,-1).bmm(vf)
    rms=((of-ref).square().mean((1,2))/ref.square().mean((1,2)).clamp_min(1e-30)).sqrt().tolist()
    layers=[i for i in range(50) if i not in (45,48,49)]
    call=STATS["calls"]
    record=dict(call=call,request=(call-1)//188,step=((call-1)//47)%4,layer=layers[(call-1)%47],
                rows=rows_n,heads=heads,rms=rms,max_rms=max(rms),bounds=[LAST_BOUNDS[i] for i in heads],
                fallback_heads=slow,seconds=time.perf_counter()-started)
    with (Path(folder)/"attention-reference.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
    print("[h3reference] "+json.dumps(record),flush=True)


_CALL_LOCK=threading.Lock()
@torch.compiler.disable
def direct_max_free_attention(query,key,value,scale,original):
    # Protect persistent staging and native head selection against reentrancy.
    # Other unrelated device computation still requires the caller's usual
    # single-worker/stream synchronization contract.
    with _CALL_LOCK:
        return _direct_max_free_attention_impl(query,key,value,scale,original)


def install(attn=None) -> bool:
    """Patch vllm_gaudi.omni.attention.max_free_attention (H3 max-free layers only)."""
    if attn is None:
        import vllm_gaudi.omni.attention as attn
    if getattr(attn, "_h3direct_installed", False):
        return True
    original = attn.max_free_attention

    def patched(query, key, value, scale, chunk=attn.HPU_SDPA_MAX_FREE_CHUNK):
        return direct_max_free_attention(query, key, value, scale,
                                         lambda q, k, v, s: original(q, k, v, s, chunk))
    attn.max_free_attention = patched
    attn._h3direct_installed = True
    return True
