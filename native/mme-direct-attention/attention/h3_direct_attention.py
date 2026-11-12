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
# Key-capacity buckets. H3 attention length = 37296 video + 414 audio + text tokens (1344x768, 124 frames).
#   37760: text <= 50 tokens (bit-identical to the original single-length plan)
#   then one bucket every 512 keys (38272 ... 41344: text <= 562, 1074, 1586, 2098, 2610, 3122, 3634 tokens),
#   so a length wastes at most about 512 keys (about 1.4%); 41344 has the largest SRAM fit (first quarter 10368).
# Every bucket keeps its last quarter narrower than the first, so the TPC programs' structure does not depend
# on the valid length and context_quad_set_valid can rewrite them in place (41472 with equal quarters failed
# at L = 41472: the exp setup deduplicates equal column counts, so the program size changed).
# H3DIRECT_KEYS=auto (default) uses the smallest bucket that holds a call's length and rebuilds the plan when
# that changes (the result for a given length never depends on earlier calls); a number pins one bucket.
BUCKETS = (37760, 38272, 38784, 39296, 39808, 40320, 40832, 41344)
KEYS_SETTING = os.environ.get("H3DIRECT_KEYS", "auto")
if KEYS_SETTING != "auto" and int(KEYS_SETTING) not in BUCKETS:
    raise ValueError(f"H3DIRECT_KEYS must be auto or one of {BUCKETS}, got {KEYS_SETTING}")
VALID, DIM = 37750, 128         # VALID: a plan's initial valid length (any in-range length works afterwards)
GROUP = 16                      # 128-query blocks per native group
PARTS, QUANTUM = 4, 128


def geometry(keys: int) -> tuple[int, int, int]:
    """(first key-quarter width, query blocks (whole groups), smallest valid length) for a key capacity."""
    first = (keys + PARTS * QUANTUM - 1) // (PARTS * QUANTUM) * QUANTUM
    blocks = (keys + GROUP * 128 - 1) // (GROUP * 128) * GROUP
    return first, blocks, 3 * first + 1          # the last key quarter must hold at least one valid key


def select_bucket(length: int) -> int | None:
    allowed = BUCKETS if KEYS_SETTING == "auto" else (int(KEYS_SETTING),)
    for keys in allowed:
        if geometry(keys)[2] <= length <= keys:
            return keys
    return None


KEYS = BUCKETS[0] if KEYS_SETTING == "auto" else int(KEYS_SETTING)   # compatibility (old tests)
FIRST, BLOCKS, MIN_VALID = geometry(KEYS)
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
    def __init__(self, relaxed: bool, heads: int, keys: int, valid0: int | None = None):
        self.heads=heads
        self.keys = keys
        self.first, self.blocks, self.min_valid = geometry(keys)
        if valid0 is None:
            valid0 = VALID if self.min_valid <= VALID <= keys else keys
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
        for h, width in enumerate((self.first, keys - 3 * self.first)):
            assert setter(h, *dargs(load(f"g-m512n{width}k128", True))) == 0
            assert setter(h + 2, *dargs(load(f"g-m512n128k{width}", False))) == 0

        dev = "hpu"
        self.qh = torch.zeros((heads, self.blocks * 128, DIM), dtype=torch.bfloat16, device=dev)
        self.kth = torch.zeros((heads, DIM * keys), dtype=torch.bfloat16, device=dev)
        self.vh = torch.zeros((heads, keys, DIM), dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros((self.blocks, 128, DIM), dtype=torch.bfloat16, device=dev)
        self.stats = torch.zeros((self.blocks, STAT_BYTES // 8, 2), dtype=torch.float32, device=dev)
        torch.hpu.synchronize()

        info = (u64 * 5)()
        assert api("context_info", [u, up])(0, info) == 0
        global _RUNTIME_READY
        if not _RUNTIME_READY:   # process-level: the private stream and the LUT upload happen once
            assert api("context_scal_nop", [C.c_int])(int(info[0])) == 0
        base = info[1] + 0x200000
        dp = _data_ptr
        assert api("context_quad_set_wide_kv", [u64, u64])(dp(self.kth), dp(self.vh)) == 0
        L = (u64 * 10)()
        assert api("context_quad_layout", [u64, u, u, up])(base, keys, GROUP, L) == 0 and L[9] <= info[1] + info[2]
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
        if not _RUNTIME_READY:
            assert api("context_attention_set_lut", [C.c_char_p])(str(ROOT / "build" / LUT).encode()) == 0
            _RUNTIME_READY = True
        exp_bin = str(ROOT / "build" / (EXP_KERNEL + ".bin")).encode()
        norm_bin = str(ROOT / "build" / NORM).encode()
        assert api("context_quad_prepare", [u64, u, u, u, u, C.c_char_p, C.c_char_p])(
            base, keys, valid0, GROUP, 24, exp_bin, norm_bin) == 0
        A = (u64 * 5)()
        assert api("context_quad_layout_alt", [u64, u, u, up])(base, keys, GROUP, A) == 0 and A[4] <= info[1] + info[2]
        assert api("context_quad_prepare_slot", [u, u64, u, u, u, u, C.c_char_p, C.c_char_p])(
            1, base, keys, valid0, GROUP, 24, exp_bin, norm_bin) == 0
        self.valid = valid0
        self.staged = valid0           # rows [0, staged) of qh/vh (and K columns) may hold data
        self._release = api("context_quad_release", [])
        self._set_valid = api("context_quad_set_valid", [u])
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
        self.head_bytes=self.blocks*128*DIM*2
        # Continuous streaming (H3DIRECT_STREAM=1): groups of all heads form one
        # sequence; global index parity selects the SRAM slot, so a head may start
        # in either slot. Precompute both parities per head.
        self.select_head_any=api("context_quad_select_head_any", [u])
        self.stream_tr=[[[None]*(self.blocks//GROUP) for _ in range(heads)] for _ in range(2)]
        for parity in range(2):
            for head in range(heads):
                for group in range(self.blocks // GROUP):
                    sq, so, ss = slots[(parity+group) % 2]
                    inp = transfer([(dp(self.qh)+head*self.head_bytes+group*GROUP*32768, sq, GROUP*32768)])
                    ret = transfer([(so, dp(self.out)+group*GROUP*32768, GROUP*32768),
                                    (ss, dp(self.stats)+group*GROUP*STAT_BYTES, GROUP*STAT_BYTES)])
                    self.stream_tr[parity][head][group]=(inp,ret)
        self.transfers = [[] for _ in range(heads)]
        for head in range(heads):
            for group in range(self.blocks // GROUP):
                sq, so, ss = slots[group % 2]
                inp = transfer([(dp(self.qh)+head*self.head_bytes+group*GROUP*32768, sq, GROUP*32768)])
                ret = transfer([(so, dp(self.out)+group*GROUP*32768, GROUP*32768),
                                (ss, dp(self.stats)+group*GROUP*STAT_BYTES, GROUP*STAT_BYTES)])
                self.transfers[head].append((inp,ret))

    def release(self):
        """Free the native plan (all ECBs back to their pools) and the staging tensors."""
        status = self._release()
        if status:
            raise RuntimeError(f"context_quad_release failed: {status}")
        for name in ("qh", "kth", "vh", "out", "stats"):
            setattr(self, name, None)

    def set_valid(self, valid: int) -> bool:
        """Switch the prepared TPC programs to another valid key count (nothing may be queued).

        The native update is all-or-nothing (a failed write restores the old code), but this object still forgets
        its length on any failure: the next call of any length goes to the native side, which is a no-op when the
        programs already match. On False the caller must not run this plan at `valid`; it prepares a new plan."""
        if self.valid is not None and valid == self.valid:
            return True
        status = self._set_valid(valid)
        if status:
            self.valid = None
            print(f"[h3direct] set_valid({valid}) failed: {status}; this plan is not used at that length", flush=True)
            return False
        self.valid = valid
        return True

    def clear_tail(self, valid: int):
        """Zero Q/V rows and K columns in [valid, staged): padded keys must score exactly 0 and padded values must
        be finite zeros (0 * stale NaN would poison the PV product)."""
        if self.staged > valid:
            hi = self.staged
            self.qh[:, valid:hi].zero_()
            self.vh[:, valid:min(hi, self.keys)].zero_()
            for part in range(PARTS):
                start = part * self.first; width = min(self.first, self.keys - start)
                lo, top = max(valid - start, 0), min(hi - start, width)
                if lo < top:
                    self.kth[:, start * DIM:(start + width) * DIM].view(-1, DIM, width)[:, :, lo:top].zero_()
        self.staged = valid

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


    def run_heads_stream(self, heads, output_base: int):
        """All listed heads as one cross-group sequence (two groups in flight
        across head boundaries). Same fences as within a head: every group after
        the first waits for its predecessor's two last PVs before reusing S."""
        groups=self.blocks//GROUP
        seq=[]
        for head in heads:
            parity=len(seq)&1
            for g in range(groups):
                inp,ret=self.stream_tr[parity][head][g]
                ret[2][0]=output_base+head*self.head_bytes+g*GROUP*32768
                seq.append((head,inp,ret))
        n=len(seq)
        if not n:return
        def start(i):
            head,inp,_=seq[i]
            assert self.select_head_any(head)==0
            return inp
        assert self.dma(*start(0))==0
        assert self.submit2(0,0)==0
        if n>1:
            assert self.dma(*start(1))==0
            assert self.submit2(1,1)==0
        for i in range(n):
            status=self.wait()
            if status:
                raise RuntimeError(f"direct attention stream group {i} failed: {status}")
            assert self.dma_enqueue(*seq[i][2])==0
            if i+2<n:
                assert self.dma_enqueue(*seq[i+2][1])==0
            assert self.dma_sync()==0
            if i+2<n:
                assert self.select_head_any(seq[i+2][0])==0
                assert self.submit2((i+2)&1,1)==0


_PLAN: _Plan | None = None
_RUNTIME_READY = False   # private SCAL stream and LUT initialised (once per process)
STATS = {"calls": 0, "direct_heads": 0, "fallback_heads": 0, "direct_s": 0.0, "plan_switches": 0, "plan_prepare_s": 0.0,
         "gates_s":0.0,"pack_enqueue_s":0.0,"head_staging_wait_s":0.0,"head_compute_s":0.0}


def _plan(heads: int = 1, keys: int = KEYS, valid0: int | None = None, rebuild: bool = False) -> _Plan:
    """The plan for this bucket; another bucket (or rebuild=True) releases the current plan and prepares a new
    one, prepared for valid0 when given."""
    global _PLAN
    if _PLAN is not None and (_PLAN.keys != keys or rebuild):
        torch.hpu.synchronize()
        _PLAN.release()
        _PLAN = None
        STATS["plan_switches"] += 1
    if _PLAN is None:
        t = time.perf_counter()
        _PLAN = _Plan(relaxed=os.environ.get("H3DIRECT_RELAXED", "0") == "1", heads=heads, keys=keys, valid0=valid0)
        STATS["plan_prepare_s"] += time.perf_counter() - t
        print(f"[h3direct] plan prepared keys={keys} blocks={_PLAN.blocks} in {time.perf_counter() - t:.2f} s",
              flush=True)
    return _PLAN


def head_gates(qs: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_factor: float = 1.0,
               valid: int = VALID) -> list[bool]:
    """Fixed-0 gates of batch_attention_probe.py per head. qs/key/value [H, L, D] (qs already scaled)."""
    kmax = key.float().norm(dim=-1).amax(-1)                      # [H]
    upper = (qs.float().norm(dim=-1).amax(-1) * kmax * score_factor)              # [H] = max_i |q_i| * max_j |k_j|
    vmax = value.abs().amax(dim=(-1, -2)).float()                 # [H]
    upper, vmax = upper.cpu().tolist(), vmax.cpu().tolist()
    shift = EXP_SHIFT * math.log(2.0)
    ok = []
    for b, vm in zip(upper, vmax):
        good = math.isfinite(b) and math.isfinite(vm) and b <= 80
        good = good and -b - shift - .25 > -86
        good = good and b - shift + .25 + math.log(valid) + math.log(max(1., vm)) < 86
        good = good and math.log(valid) - b - shift - .25 > math.log(1e-30)
        ok.append(good)
    return ok


def _stage_valid(plan, Lq) -> bool:
    """Point the plan at Lq valid keys and clear stale tails. Our programs are never queued between calls."""
    if not plan.set_valid(Lq):
        return False
    plan.clear_tail(Lq)
    return True


def _direct_max_free_attention_impl(query, key, value, scale, original, exact):
    """Drop-in for max_free_attention(query, key, value, scale) -> [B, H, L, D]."""
    B, H, Lq, D = query.shape
    keys = select_bucket(Lq)
    if (B != 1 or D != DIM or keys is None or key.shape != query.shape or value.shape != query.shape
            or query.dtype != torch.bfloat16 or query.device.type != "hpu"
            or key.dtype!=query.dtype or value.dtype!=query.dtype
            or key.device!=query.device or value.device!=query.device or not 1<=H<=64
            or query.requires_grad or key.requires_grad or value.requires_grad):
        return original(query, key, value, scale)
    if _PLAN is not None and _PLAN.heads!=H:
        return original(query,key,value,scale)
    if POW2_Q and not math.isclose(float(scale), DIM**-.5, rel_tol=0, abs_tol=1e-12):
        return original(query,key,value,scale)
    plan = _plan(H, keys)
    if not _stage_valid(plan, Lq):
        # In-place rewrite refused (program layout changed): prepare the plan at this length instead.
        plan = _plan(H, keys, valid0=Lq, rebuild=True)
        if not _stage_valid(plan, Lq):
            return exact(query, key, value, scale)
    if POW2_Q and os.environ.get("H3DIRECT_FAST_GATES","0")=="1":
        return _fast_gates_impl(plan,query,key,value,scale,exact)
    t0 = time.perf_counter()
    qs = query[0] * (INPUT_Q_SCALE if POW2_Q else scale)                                          # [H, L, D] bf16, as exported
    k, v = key[0], value[0]
    gates = head_gates(qs, k, v, SCORE_FACTOR, Lq)
    t_pack=time.perf_counter();STATS["gates_s"]+=t_pack-t0
    # Stage all heads at once. Static head banks bind K/V into these owned,
    # persistent tensors; Q/O DMA runs without intervening framework work.
    plan.qh[:,:Lq].copy_(qs)
    plan.vh[:,:Lq].copy_(v)
    # Transpose straight into each persistent K quarter. Padding stays zero;
    # avoid a full padded K allocation, four temporaries and the concat copy.
    for part in range(PARTS):
        start=part*plan.first;width=min(plan.first,plan.keys-start);valid_width=min(width,Lq-start)
        target=plan.kth[:,start*DIM:(start+width)*DIM].view(H,DIM,width)
        target[:,:,:valid_width].copy_(k[:,start:start+valid_width].transpose(1,2))
    padded=torch.empty((H,plan.blocks*128,DIM),dtype=query.dtype,device=query.device)
    result=padded[:,:Lq]
    STATS["pack_enqueue_s"]+=time.perf_counter()-t_pack
    t_stage=time.perf_counter();torch.hpu.synchronize()
    STATS["head_staging_wait_s"]+=time.perf_counter()-t_stage
    output_base=plan._data_ptr(padded)
    slow = [h for h in range(H) if not gates[h]]
    if os.environ.get("H3DIRECT_STREAM","0")=="1":
        t_compute=time.perf_counter()
        plan.run_heads_stream([h for h in range(H) if gates[h]],output_base)
        STATS["head_compute_s"]+=time.perf_counter()-t_compute
    else:
      for h in range(H):
        if not gates[h]:continue
        t_compute=time.perf_counter()
        plan.run_head(h,output_base+h*plan.head_bytes)
        STATS["head_compute_s"]+=time.perf_counter()-t_compute
    if slow:
        idx = torch.tensor(slow, device=query.device)
        result[idx] = exact(query[:, idx], key[:, idx], value[:, idx], scale)[0]
    torch.hpu.synchronize()
    STATS["calls"] += 1
    STATS["direct_heads"] += H - len(slow)
    STATS["fallback_heads"] += len(slow)
    elapsed=time.perf_counter()-t0
    STATS["direct_s"] += elapsed
    if os.environ.get("H3DIRECT_LOG")=="1":
        print("[h3direct] "+json.dumps(dict(call=STATS["calls"],heads=H,valid=Lq,direct_heads=H-len(slow),
              fallback_heads=slow,pow2_q=POW2_Q,wall_s=round(elapsed,6))),flush=True)
    return result.unsqueeze(0)


def _gate_decisions(upper, vmax, finite, valid=VALID):
    shift = EXP_SHIFT * math.log(2.0)
    ok = []
    for b, vm, fin in zip(upper, vmax, finite):
        good = fin and math.isfinite(b) and math.isfinite(vm) and b <= 80
        good = good and -b - shift - .25 > -86
        good = good and b - shift + .25 + math.log(valid) + math.log(max(1., vm)) < 86
        good = good and math.log(valid) - b - shift - .25 > math.log(1e-30)
        ok.append(good)
    return ok


def _gate_tensor(q, k, v, q_scale: float, score_factor: float):
    """[3, H]: score upper bound, |V| max, NaN/Inf detector (sums propagate)."""
    qn = torch.linalg.vector_norm(q, dim=-1, dtype=torch.float32)
    kn = torch.linalg.vector_norm(k, dim=-1, dtype=torch.float32)
    vn = torch.linalg.vector_norm(v, dim=-1, dtype=torch.float32)
    upper = qn.amax(-1) * q_scale * kn.amax(-1) * score_factor
    vmax = v.abs().amax(dim=(-1, -2)).float()
    check = qn.sum(-1) + kn.sum(-1) + vn.sum(-1)
    return torch.stack([upper, vmax, check])


_GATE_COMPILED = None


def _gate_fn():
    global _GATE_COMPILED
    if os.environ.get("H3DIRECT_COMPILED_GATES", "0") != "1":
        return _gate_tensor
    if _GATE_COMPILED is None:
        _GATE_COMPILED = torch.compile(_gate_tensor, backend="hpu_backend", dynamic=False)
    return _GATE_COMPILED


def _fast_gates_impl(plan, query, key, value, scale, exact):
    """Opt-in (H3DIRECT_FAST_GATES=1, POW2Q only). Same gate formulas as
    head_gates, but: norms accumulate in FP32 without materialising FP32 copies;
    |Q*2^-4| = 2^-4*|Q| exactly; all staging copies are enqueued before the one
    readback, so gate reductions and staging overlap on the device. NaN/Inf in
    any of Q/K/V is detected by per-head sums of row norms (sums propagate
    NaN/Inf), independently of amax NaN semantics."""
    B, H, Lq, D = query.shape
    t0 = time.perf_counter()
    q, k, v = query[0], key[0], value[0]
    gate_vals = _gate_fn()(q, k, v, INPUT_Q_SCALE, SCORE_FACTOR)          # [3, H], read later
    t_pack = time.perf_counter(); STATS["gates_s"] += t_pack - t0
    torch.mul(q, INPUT_Q_SCALE, out=plan.qh[:, :Lq])                       # exact 2^-4, same as qs
    plan.vh[:, :Lq].copy_(v)
    for part in range(PARTS):
        start=part*plan.first;width=min(plan.first,plan.keys-start);valid_width=min(width,Lq-start)
        target=plan.kth[:,start*DIM:(start+width)*DIM].view(H,DIM,width)
        target[:,:,:valid_width].copy_(k[:,start:start+valid_width].transpose(1,2))
    padded=torch.empty((H,plan.blocks*128,DIM),dtype=query.dtype,device=query.device)
    result=padded[:,:Lq]
    STATS["pack_enqueue_s"]+=time.perf_counter()-t_pack
    t_stage=time.perf_counter()
    upper_l, vmax_l, check_l = gate_vals.cpu().tolist()                  # the one readback
    torch.hpu.synchronize()
    STATS["head_staging_wait_s"]+=time.perf_counter()-t_stage
    gates = _gate_decisions(upper_l, vmax_l, [math.isfinite(c) for c in check_l], Lq)
    output_base=plan._data_ptr(padded)
    slow = [h for h in range(H) if not gates[h]]
    t_compute=time.perf_counter()
    if os.environ.get("H3DIRECT_STREAM","0")=="1":
        plan.run_heads_stream([h for h in range(H) if gates[h]],output_base)
    else:
        for h in range(H):
            if gates[h]: plan.run_head(h,output_base+h*plan.head_bytes)
    STATS["head_compute_s"]+=time.perf_counter()-t_compute
    if slow:
        idx = torch.tensor(slow, device=query.device)
        result[idx] = exact(query[:, idx], key[:, idx], value[:, idx], scale)[0]
    torch.hpu.synchronize()
    STATS["calls"] += 1
    STATS["direct_heads"] += H - len(slow)
    STATS["fallback_heads"] += len(slow)
    elapsed=time.perf_counter()-t0
    STATS["direct_s"] += elapsed
    if os.environ.get("H3DIRECT_LOG")=="1":
        print("[h3direct] "+json.dumps(dict(call=STATS["calls"],heads=H,valid=Lq,direct_heads=H-len(slow),
              fallback_heads=slow,pow2_q=POW2_Q,fast_gates=True,wall_s=round(elapsed,6))),flush=True)
    return result.unsqueeze(0)


def _exact_attention(query, key, value, scale):
    """Row-max softmax attention [B, H, L, D] for rejected heads (same algorithm as the fixed vllm-gaudi
    exact_attention): FusedSDPA with stats, aligned key head + tail merged exactly."""
    import vllm_gaudi.omni.attention as attn
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    if query.device.type != "hpu":
        return torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=scale)
    heads, factor = query.shape[1], int(os.environ.get("PT_HPU_SDPA_HEAD_FACTOR", "1") or 1)
    if heads < factor:  # FusedSDPA fails to compile with fewer heads than PT_HPU_SDPA_HEAD_FACTOR
        widen = lambda t: t.repeat(1, -(-factor // heads), 1, 1)[:, :factor]  # noqa: E731
        return _exact_attention(widen(query), widen(key), widen(value), scale)[:, :heads]
    split = attn.kv_split_point(key.shape[-2])
    if split:
        o1, m1, l1 = attn._sdpa_with_stats(query, key[..., :split, :].contiguous(), value[..., :split, :].contiguous(), scale)
        o2, m2, l2 = attn._sdpa_with_stats(query, key[..., split:, :].contiguous(), value[..., split:, :].contiguous(), scale)
        return attn.merge_kv_split(o1, m1, l1, o2, m2, l2)
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    return FusedSDPA.apply(query, key, value, None, 0.0, False, scale, attn.HPU_SDPA_SOFTMAX_MODE, True)


_CALL_LOCK=threading.Lock()
@torch.compiler.disable
def direct_max_free_attention(query,key,value,scale,original,exact=None):
    # Protect persistent staging and native head selection against reentrancy.
    # Other unrelated device computation still requires the caller's usual
    # single-worker/stream synchronization contract.
    with _CALL_LOCK:
        return _direct_max_free_attention_impl(query,key,value,scale,original,exact or _exact_attention)


def install(attn=None) -> bool:
    """Patch vllm_gaudi.omni.attention.max_free_attention (H3 max-free layers only)."""
    if attn is None:
        import vllm_gaudi.omni.attention as attn
    if getattr(attn, "_h3direct_installed", False):
        return True
    original = attn.max_free_attention
    exact = getattr(attn, "exact_attention", None) or _exact_attention

    def patched(query, key, value, scale, chunk=attn.HPU_SDPA_MAX_FREE_CHUNK):
        return direct_max_free_attention(query, key, value, scale,
                                         lambda q, k, v, s: original(q, k, v, s, chunk), exact)
    attn.max_free_attention = patched
    attn._h3direct_installed = True
    return True
