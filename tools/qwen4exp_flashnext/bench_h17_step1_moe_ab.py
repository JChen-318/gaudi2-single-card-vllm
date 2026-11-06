# SPDX-License-Identifier: Apache-2.0
"""H17 Phase 2 / Step 1: direct HPU A/B of ``qwen4_exp.Qwen4ExpSparseMoeBlock``.

The Phase-1 unification replaced the production decode MoE with the chain the
vendor ``qwen4exp/model.py`` already had.  This benchmark measures the exact
Before -> After delta on the *production class*, with real layer-0 weights and
real layer-0 geometry (hidden=2560, E=512, top_k=10, moe_inter=640):

  Before (pre-Phase-1, source copied verbatim from bc99e66):
    router        F.linear -> torch.topk -> softmax -> BF16/int32
    routed MoE    x_slots = hidden.index_select(repeat_interleave(arange))
                  -> gate GEMV -> up GEMV -> F.silu * up -> down GEMV
                  -> (down * w).view(T, K, H).sum(1)
    shared        stock Qwen2MoeMLP (gate_up -> SiluAndMul -> down
                  -> sigmoid(shared_gate) * out)

  After (Phase-1 unified ``_grouped_swiglu`` + ``shared_expert_forward_h15``):
    router        F.linear -> fused custom_qwen4exp_moe_router_topk
    routed MoE    grouped_gemv: x = hidden [T, H] (H10, no replication)
                  -> swiglu_gemv_slots (fused gate+up+silu*mul)
                  -> down GEMV -> down_epilogue (fused weighted top-k sum)
    shared        merged [2I+1, H] GEMM -> fused shared_swiglu
                  -> down GEMM (3 launches)

Also timed: an "H15 off" intermediate (``VLLM_QWEN4EXP_MOE_H15=0`` semantics,
applied at the Python switch only: H10 un-replicated x, two GEMVs, legacy
router, stock shared) so the H10 and H15 halves of the delta are separable.
In the JSON it is keyed ``h10_only``.

Accuracy (same weights, same inputs, no tolerances on the routed path):

  * routed MoE ``After`` vs the exact chain (standalone gate/up GEMV BF16
    outputs -> CPU silu*mul -> down GEMV -> BF16 weighted sum) must be
    100% bit-identical (0 differing BF16 elements), per T.
  * router ids/weights vs the legacy torch.topk chain: weights bit-identical,
    ids identical (a differing id is only acceptable if it is an exact BF16
    logit tie, reported as such).
  * full block ``After`` vs a CPU oracle (same weights, stock path) and vs
    ``Before`` on the HPU: cosine >= 0.99999.

Timing follows the H14/H11 methodology: ``time_sync`` synchronizes around
every call (attribution, launch-count sensitive) and ``time_batched`` queues
``--reps`` steps back to back with one synchronize after the loop (the
decode-step model).  In lazy mode ``mark_step()`` is issued at every step
boundary, as in ``bench_layer_e2e.py``.

Run (card must be free; eager first, then lazy -- separate processes):

  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h17_step1_moe_ab.py \
      --json tools/qwen4exp_flashnext/bench_h17_step1_moe_eager.json
  PT_HPU_LAZY_MODE=1 .venv/bin/python tools/qwen4exp_flashnext/bench_h17_step1_moe_ab.py \
      --json tools/qwen4exp_flashnext/bench_h17_step1_moe_lazy.json

  ... --check-only   accuracy only, no timing
  ... --tokens 1     one decode size only
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
EXT = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
    "hpu_qwen4exp_nvfp4_dequant_pt2*.so"), None)
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.0.mlp"

# These must be set before torch / habana_frameworks are imported.
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
# The A/B is about the H15 switch: force it on for the "after" chain.  The
# "before" chain below is an explicit copy of the pre-Phase-1 source, so it
# does not read this flag.
os.environ["VLLM_QWEN4EXP_MOE_H15"] = "1"

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import habana_frameworks.torch.core  # noqa: E402,F401

from vllm_gaudi.models.qwen4_exp import (  # noqa: E402
    Qwen4ExpSparseMoeBlock,
    _grouped_swiglu,
    track_c_enabled,
)
from vllm_gaudi.models.qwen4exp.packed_moe import (  # noqa: E402
    down_epilogue,
    gemv_slots,
    grouped_gemv,
    moe_h15_enabled,
    route_topk,
    shared_expert_forward_h15,
    swiglu_gemv_slots,
)

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
MOE_INTER = 640
DIST_READY = False
_HANDLES: dict = {}
_HEADERS: dict = {}
_WEIGHT_MAP: dict | None = None
_SAFETENSORS_DTYPE = {
    "BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8,
    "I8": torch.int8, "F16": torch.float16, "I32": torch.int32,
}


# ---------------------------------------------------------------------------
# checkpoint loading (same path as test_moe_block_hpu.py)
# ---------------------------------------------------------------------------
def _weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        _WEIGHT_MAP = json.loads(
            (CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    return _WEIGHT_MAP


def _load(name: str) -> torch.Tensor:
    """Read exactly this tensor's byte range from its shard.

    ``safetensors.safe_open(...).get_tensor`` reads the whole shard per call
    on this host (the layer-0 planes live in all 21 x ~5 GB shards, so one
    block load costs ~105 GB of reads). The shard header already carries each
    tensor's data_offsets, so seek + read of the range is equivalent and
    bit-identical (checked against safe_open for a layer-0 gate plane), and
    turns the load into ~1.4 GB of reads.
    """
    shard_name = _weight_map()[name]
    if shard_name not in _HEADERS:
        path = CHECKPOINT / "tensors" / shard_name
        with open(path, "rb") as f:
            header_bytes = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_bytes))
        _HEADERS[shard_name] = (header, 8 + header_bytes, path)
    header, data_base, path = _HEADERS[shard_name]
    info = header[name]
    handle = _HANDLES.get(shard_name)
    if handle is None:
        handle = open(path, "rb")
        _HANDLES[shard_name] = handle
    start, end = info["data_offsets"]
    handle.seek(data_base + start)
    buf = handle.read(end - start)
    dtype = info["dtype"]
    if dtype == "F8_E4M3":
        tensor = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(torch.float8_e4m3fn)
    else:
        tensor = torch.frombuffer(bytearray(buf), dtype=_SAFETENSORS_DTYPE[dtype])
    return tensor.reshape(info["shape"])


def _build() -> tuple[Qwen4ExpSparseMoeBlock, object]:
    import tempfile

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
    assert cfg.hidden_size == HIDDEN and cfg.num_experts == EXPERTS, (cfg.hidden_size, cfg.num_experts)
    assert cfg.num_experts_per_tok == TOP_K, cfg.num_experts_per_tok
    assert cfg.moe_intermediate_size == MOE_INTER, cfg.moe_intermediate_size

    dummy = Path(tempfile.mkdtemp()) / "dummy-model"
    dummy.mkdir(parents=True, exist_ok=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 262144,
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=8192,
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)

    global DIST_READY
    if not DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        kept = set_current_vllm_config(vllm_config)
        kept.__enter__()
        initialize_model_parallel(1, 1)
        DIST_READY = True

    # The engine builds the model under set_default_torch_dtype(model dtype);
    # do the same, so the Linear parameters are BF16 like production (a
    # float32 block would fail the BF16 router GEMM and disable H15 shared).
    with set_default_torch_dtype(torch.bfloat16):
        block = Qwen4ExpSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")
    return block, cfg


def _weights(cfg) -> list[tuple[str, torch.Tensor]]:
    """Block-relative names, exactly as test_moe_block_hpu.py feeds them."""
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    out: list[tuple[str, torch.Tensor]] = [
        ("gate.weight", _load(f"{LAYER}.gate.weight").contiguous()),
        ("shared_expert_gate.weight", _load(f"{LAYER}.shared_expert_gate.weight").contiguous()),
    ]
    # The qwen4exp_plain loader (``iter_bf16_weights``) dequantizes every
    # MXFP8 shared-expert plane to BF16; do the same so the block the H15
    # path sees is the one production sees (test_moe_block_hpu.py passes the
    # down plane raw F8; that is not the served weight).
    base = f"{LAYER}.shared_expert.down_proj"
    out.append(("shared_expert.down_proj.weight",
                dequant_mxfp8(_load(base + ".weight"), _load(base + ".weight_scale"))))
    for shard_id, kind in enumerate(("gate_proj", "up_proj")):
        base = f"{LAYER}.shared_expert.{kind}"
        weight = dequant_mxfp8(_load(base + ".weight"), _load(base + ".weight_scale"))
        weight.shard_id = shard_id
        out.append(("shared_expert.gate_up_proj.weight", weight))
    for expert in range(cfg.num_experts):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            base = f"{LAYER}.experts.{expert}.{kind}.weight"
            out.append((f"experts.{expert}.{kind}.weight", _load(base).contiguous()))
            out.append((f"experts.{expert}.{kind}.weight_scale",
                        _load(base + "_scale").view(torch.uint8)))
            out.append((f"experts.{expert}.{kind}.weight_scale_2",
                        _load(base + "_scale_2")))
    return out


# ---------------------------------------------------------------------------
# timing helpers (H14/H11 methodology; mark_step at lazy step boundaries)
# ---------------------------------------------------------------------------
def _is_lazy() -> bool:
    return os.environ.get("PT_HPU_LAZY_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


def _mark_step() -> None:
    if not _is_lazy():
        return
    try:
        import habana_frameworks.torch.core as _htcore

        _htcore.mark_step()
    except Exception:
        pass


def _sync_step() -> None:
    _mark_step()
    torch.hpu.synchronize()


def _median(xs):
    return statistics.median(xs)


def _checksum(out) -> float:
    if out is None:
        return 0.0
    if isinstance(out, torch.Tensor):
        return float(out.float().sum().item())
    if isinstance(out, (tuple, list)):
        return sum(_checksum(o) for o in out)
    return 0.0


def time_sync(fn, warmup: int, samples: int, raw: list | None = None) -> float:
    """Median ms/call, synchronize around every call (launch attribution)."""
    check = 0.0
    for _ in range(warmup):
        with torch.no_grad():
            check += _checksum(fn())
        _sync_step()
    times = []
    for _ in range(samples):
        _sync_step()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = fn()
        _sync_step()
        times.append((time.perf_counter() - t0) * 1e3)
        check += _checksum(out)
    if check != check or abs(check) == float("inf"):
        raise SystemExit(f"non-finite checksum {check}")
    if raw is not None:
        raw.extend(times)
    return _median(times)


def time_batched(fn, reps: int, samples: int, raw: list | None = None) -> float:
    """Unsynchronized ms/call: ``reps`` steps queued, one sync after the loop."""
    acc = torch.zeros((), dtype=torch.float32, device="hpu")

    def one_rep():
        nonlocal acc
        with torch.no_grad():
            out = fn()
        if isinstance(out, (tuple, list)):
            out = out[0]  # router returns (ids, weights); ids is a tensor
        acc = acc + out.float().sum()
        _mark_step()

    for _ in range(2):
        one_rep()
    torch.hpu.synchronize()
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        for _ in range(reps):
            one_rep()
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / reps)
    check = float(acc.float().item())
    if check != check or abs(check) == float("inf"):
        raise SystemExit(f"non-finite checksum {check}")
    if raw is not None:
        raw.extend(times)
    return _median(times)


def time_chained(fn, inputs: list, samples: int, raw: list | None = None) -> float:
    """Per-call ms for ``len(inputs)`` blocks built into ONE lazy graph.

    A production decode step marks once after the whole stack of layers, so a
    per-block mark_step over-charges graph construction.  Here ``inputs`` are
    distinct tensors (no CSE of identical calls), all ``len(inputs)`` blocks
    are enqueued and mark_step + synchronize happen once, then the elapsed
    time is divided by the number of blocks.  Only used in lazy mode.
    """
    def one_pass():
        with torch.no_grad():
            out = None
            for x in inputs:
                out = fn(x)
        _sync_step()
        return out

    one_pass()
    one_pass()
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        out = one_pass()
        times.append((time.perf_counter() - t0) * 1e3 / len(inputs))
    if raw is not None:
        raw.extend(times)
    return _median(times)


@contextlib.contextmanager
def h15_off():
    """Disable only the H15 Python switch (H10 chain still engages)."""
    import vllm_gaudi.models.qwen4exp.packed_moe as pm

    old = pm.moe_h15_enabled
    pm.moe_h15_enabled = lambda: False
    try:
        yield
    finally:
        pm.moe_h15_enabled = old


# ---------------------------------------------------------------------------
# Before / After compute
# ---------------------------------------------------------------------------
def route_before(block, hidden):
    """Pre-Phase-1 router: F.linear -> torch.topk -> softmax -> BF16/int32."""
    logits = F.linear(hidden, block.gate.weight)
    values, indices = torch.topk(logits, block.top_k, dim=-1)
    weights = torch.softmax(values.float(), dim=-1).to(hidden.dtype)
    return indices.to(torch.int32), weights


def routed_before(block, hidden, topk_ids, topk_weights):
    """Verbatim pre-Phase-1 ``_grouped_swiglu`` (bc99e66)."""
    tokens, top_k = topk_ids.shape
    ids = topk_ids.reshape(-1).to(dtype=torch.int32).contiguous()
    token_index = torch.arange(tokens, device=hidden.device).repeat_interleave(top_k)
    shared = hidden.index_select(0, token_index)
    gate = block.experts.forward_proj("gate_proj", ids, shared)
    up = block.experts.forward_proj("up_proj", ids, shared)
    activated = F.silu(gate) * up
    down = block.experts.forward_proj("down_proj", ids, activated)
    mix = topk_weights.reshape(-1, 1).to(dtype=down.dtype)
    return (down * mix).view(tokens, top_k, -1).sum(dim=1)


def block_before(block, hidden_states):
    """Verbatim pre-Phase-1 block forward (routed loop + stock shared MLP)."""
    orig_shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, orig_shape[-1])
    topk_ids, topk_weights = route_before(block, hidden)
    out = routed_before(block, hidden, topk_ids, topk_weights)
    if block.shared_expert is not None:
        out = out + block.shared_expert(hidden)
    return out.view(orig_shape)


def block_before_served(block, hidden_states):
    """Pre-Phase-1 production block: fused router (H15 on) + legacy routed loop."""
    orig_shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, orig_shape[-1])
    topk_ids, topk_weights = route_topk(hidden, block.gate.weight, block.top_k)
    out = routed_before(block, hidden, topk_ids, topk_weights)
    if block.shared_expert is not None:
        out = out + block.shared_expert(hidden)
    return out.view(orig_shape)


def routed_after(block, hidden, topk_ids, topk_weights):
    return _grouped_swiglu(hidden, topk_ids, topk_weights, block.experts)


def shared_after(block, hidden):
    out = shared_expert_forward_h15(
        block.shared_expert, hidden, expert_gate=block.shared_expert_gate)
    if out is None:
        raise SystemExit("shared_expert_forward_h15 declined the BF16 checkpoint weights")
    return out


def block_after(block, hidden_states):
    return block(hidden_states)


def exact_routed(block, hidden, topk_ids, topk_weights):
    """Exact chain: standalone GEMVs + CPU silu*mul + BF16 weighted sum.

    This is the pre-H15 arithmetic with an exact (CPU) silu, i.e. the
    reference the fused SwiGLU GEMV and the down epilogue claim bit-equality
    against.  Returns (out, swiglu_act, slot_down).
    """
    tokens, top_k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(dtype=torch.int32).contiguous()
    gate = gemv_slots(block.experts, "gate_proj", flat, hidden)
    up = gemv_slots(block.experts, "up_proj", flat, hidden)
    act = (F.silu(gate.cpu()) * up.cpu()).to("hpu")
    slot_down = gemv_slots(block.experts, "down_proj", flat, act.contiguous())
    prod = slot_down * topk_weights.reshape(-1, 1).to(dtype=slot_down.dtype)
    out = prod.view(tokens, top_k, -1).sum(dim=1)
    return out, act, slot_down


def exact_routed_cpu(slot_down: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """CPU emulation of the down epilogue: BF16 product, F32 sum, BF16 store."""
    tokens, top_k = topk_weights.shape
    prod = (slot_down.float().cpu() * topk_weights.float().cpu().reshape(-1, 1))
    prod_bf = prod.to(torch.bfloat16).float()
    summed = prod_bf.view(tokens, top_k, -1).sum(dim=1)
    return summed.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# accuracy helpers
# ---------------------------------------------------------------------------
def _diff_bits(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    a_c, b_c = a.cpu(), b.cpu()
    if a_c.dtype == torch.bfloat16:
        return int((a_c.view(torch.uint16) != b_c.view(torch.uint16)).sum().item()), a_c.numel()
    return int((a_c != b_c).sum().item()), a_c.numel()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().cpu().reshape(1, -1), b.float().cpu().reshape(1, -1)).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a_c, b_c = a.float().cpu(), b.float().cpu()
    gap = (a_c - b_c).abs().max().item()
    return gap / max(b_c.abs().max().item(), 1e-12)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--tokens", type=str, default="1,4")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--skip-batched", action="store_true")
    args = ap.parse_args()
    tokens_list = [int(t) for t in args.tokens.split(",") if t.strip()]

    if not KERNEL.is_file() or EXT is None:
        raise SystemExit("build kernels + pytorch extension first (see bench_decode_moe.py)")
    torch.ops.load_library(str(EXT))
    ns = torch.ops.custom_op
    required = [
        "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2",
        "custom_qwen4exp_nvfp4_indexed_swiglu_gemv_bf16_gaudi2",
        "custom_qwen4exp_moe_router_topk_bf16_gaudi2",
        "custom_qwen4exp_moe_shared_swiglu_bf16_gaudi2",
        "custom_qwen4exp_moe_down_epilogue_bf16_gaudi2",
    ]
    missing = [n for n in required if not hasattr(ns, n)]
    if missing:
        raise SystemExit(f"extension is missing H15 ops: {missing}")
    if track_c_enabled():
        raise SystemExit("Track C operator is loaded: the production decode chain would not be H10/H15")
    if not moe_h15_enabled():
        raise SystemExit("VLLM_QWEN4EXP_MOE_H15 is off; force it to 1 for the After chain")

    print(f"bench_h17_step1: lazy={os.environ.get('PT_HPU_LAZY_MODE')} "
          f"device=HPU geometry=hidden{HIDDEN}/E{EXPERTS}/top{TOP_K}/I{MOE_INTER}")
    print("loading layer-0 weights ...", flush=True)
    block, cfg = _build()
    block.load_weights(iter(_weights(cfg)))
    assert block.experts.gate_proj_packed.shape == (EXPERTS, MOE_INTER, HIDDEN // 2), \
        block.experts.gate_proj_packed.shape
    assert block.experts.gate_proj_scale2.abs().sum() > 0
    print(f"loaded: gate packed {tuple(block.experts.gate_proj_packed.shape)} uint8; "
          f"shared_expert gate_up {tuple(block.shared_expert.gate_up_proj.weight.shape)} "
          f"{block.shared_expert.gate_up_proj.weight.dtype}", flush=True)

    gen = torch.Generator().manual_seed(20261105)
    hidden = {
        t: ((torch.randn(t, HIDDEN, generator=gen) * 0.5)).bfloat16()
        for t in tokens_list
    }

    # ---- CPU oracle: same weights, stock path, before the module moves -----
    cpu_ref = {}
    with torch.no_grad():
        for t in tokens_list:
            h_cpu = hidden[t].clone()
            cpu_ref[t] = {
                "full": block(h_cpu).clone(),
                "shared": block.shared_expert(h_cpu).clone(),
            }
            print(f"  cpu oracle T={t} ready", flush=True)
    block = block.to("hpu")
    hidden = {t: h.to("hpu") for t, h in hidden.items()}

    results = {
        "benchmark": "bench_h17_step1_moe_ab",
        "lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "geometry": {"hidden": HIDDEN, "experts": EXPERTS, "top_k": TOP_K, "moe_inter": MOE_INTER},
        "weights": str(CHECKPOINT / "tensors") + f" {LAYER} (real layer-0)",
        "track_c_enabled": track_c_enabled(),
        "backends": {},
        "tokens": {},
    }
    failures: list[str] = []

    # Warm the device before the first timed closure: otherwise the first
    # component of T=1 (router.before) pays the clock/recipe ramp and reads
    # ~1.7x slower than the same op in the T=4 section.
    with torch.no_grad():
        for t in tokens_list:
            for _ in range(4):
                block_before(block, hidden[t])
                block_after(block, hidden[t])
            _sync_step()

    for t in tokens_list:
        h = hidden[t]
        print(f"\n===== T={t} S={t * TOP_K} =====", flush=True)
        ids_b, w_b = route_before(block, h)
        ids_a, w_a = route_topk(h, block.gate.weight, block.top_k)
        _sync_step()

        # ---------------- accuracy --------------------------------------
        d_ids, n_ids = _diff_bits(ids_a, ids_b)
        d_w, n_w = _diff_bits(w_a, w_b)
        # A differing id is only acceptable when it is an exact BF16 logit tie.
        logits = F.linear(h, block.gate.weight)
        ref_vals, _ = torch.topk(logits, block.top_k, dim=-1)
        got_vals = torch.gather(logits.cpu(), 1, ids_a.cpu().long())
        d_vals, _ = _diff_bits(got_vals, ref_vals.cpu())
        print(f"  acc router: ids differing {d_ids}/{n_ids}, values differing {d_vals}, "
              f"weights differing {d_w}/{n_w}")
        if d_vals or d_w:
            failures.append(f"T={t} router values/weights not bit-exact (vals={d_vals} w={d_w})")

        # routed: After vs exact chain (must be bit-identical)
        exact, act_exact, slot_down = exact_routed(block, h, ids_a, w_a)
        _sync_step()
        routed_a = routed_after(block, h, ids_a, w_a)
        _sync_step()
        swiglu_a = swiglu_gemv_slots(block.experts, ids_a.reshape(-1), h)
        epi_a = down_epilogue(slot_down, w_a)
        _sync_step()
        if swiglu_a is None or epi_a is None:
            raise SystemExit("H15 routed ops are not registered")
        d_routed, n_routed = _diff_bits(routed_a, exact)
        d_swiglu, n_swiglu = _diff_bits(swiglu_a, act_exact)
        d_epi, n_epi = _diff_bits(epi_a, exact_routed_cpu(slot_down, w_a))
        print(f"  acc routed vs exact chain: differing {d_routed}/{n_routed} "
              f"({'100% bit-identical' if d_routed == 0 else 'MISMATCH'}); "
              f"swiglu {d_swiglu}/{n_swiglu}; down_epilogue {d_epi}/{n_epi}")
        if d_routed or d_swiglu or d_epi:
            failures.append(
                f"T={t} routed vs exact differing {d_routed}/{n_routed} "
                f"(swiglu {d_swiglu}, epilogue {d_epi})")

        routed_b = routed_before(block, h, ids_b, w_b)
        _sync_step()
        d_rb, _ = _diff_bits(routed_b, exact)
        cos_rb = _cos(routed_b, exact)
        print(f"  acc Before routed vs exact chain: differing {d_rb}/{n_routed} "
              f"(1-ulp HPU silu), cosine {cos_rb:.8f}")

        # shared: After (H15, 3 launches) vs Before (stock MLP) and CPU oracle
        shared_b = block.shared_expert(h)
        shared_a = shared_after(block, h)
        _sync_step()
        cos_sh_b = _cos(shared_a, shared_b)
        rel_sh_b = _rel(shared_a, shared_b)
        cos_sh_cpu = _cos(shared_a, cpu_ref[t]["shared"])
        rel_sh_cpu = _rel(shared_a, cpu_ref[t]["shared"])
        print(f"  acc shared: vs Before rel {rel_sh_b:.3e} cosine {cos_sh_b:.8f}; "
              f"vs CPU rel {rel_sh_cpu:.3e} cosine {cos_sh_cpu:.8f}")
        if not (cos_sh_b > 0.999 and cos_sh_cpu > 0.999 and rel_sh_cpu < 2e-2):
            failures.append(f"T={t} shared diverged (cos before {cos_sh_b}, cpu {cos_sh_cpu})")

        # full block: After vs Before and vs CPU oracle
        before_out = block_before(block, h)
        before_served_out = block_before_served(block, h)
        after_out = block_after(block, h)
        _sync_step()
        sum_out = routed_a + shared_a
        _sync_step()
        d_sum, n_sum = _diff_bits(after_out, sum_out)
        d_served, _ = _diff_bits(before_served_out, before_out)
        if block.experts.backends.get("decode") != "h10_h15":
            failures.append(f"T={t} decode did not take the H10/H15 chain: "
                            f"{block.experts.backends}")
        if not hasattr(block.shared_expert, "_h15_merged_weight"):
            failures.append(f"T={t} shared expert did not take the fused H15 chain")
        cos_blk_b = _cos(after_out, before_out)
        rel_blk_b = _rel(after_out, before_out)
        cos_blk_cpu = _cos(after_out, cpu_ref[t]["full"])
        rel_blk_cpu = _rel(after_out, cpu_ref[t]["full"])
        cos_b_cpu = _cos(before_out, cpu_ref[t]["full"])
        # Exact full-block chain: the bit-identical routed exact chain plus the
        # stock-MLP shared reference on CPU (same BF16 weights).  The only
        # intended difference left in After is the relocated sigmoid rounding
        # of the shared expert; the CPU oracle also differs by its bmm/F32
        # accumulation order, which is kernel-level, not a chain change.
        exact_full = exact + cpu_ref[t]["shared"].to("hpu")
        _sync_step()
        cos_blk_exact = _cos(after_out, exact_full)
        rel_blk_exact = _rel(after_out, exact_full)
        cos_b_exact = _cos(before_out, exact_full)
        print(f"  acc block: After vs Before cosine {cos_blk_b:.8f} rel {rel_blk_b:.3e}; "
              f"After vs CPU cosine {cos_blk_cpu:.8f} rel {rel_blk_cpu:.3e}; "
              f"Before vs CPU cosine {cos_b_cpu:.8f}")
        print(f"  acc block: After vs exact chain cosine {cos_blk_exact:.8f} "
              f"rel {rel_blk_exact:.3e}; Before vs exact chain cosine {cos_b_exact:.8f}")
        print(f"  acc block: After == routed_after + shared_after: differing {d_sum}/{n_sum}; "
              f"Before(H15=0) == Before(as served, fused router): differing {d_served}")
        if d_sum:
            failures.append(f"T={t} block forward differs from components ({d_sum}/{n_sum})")
        if cos_blk_exact < 0.99999:
            failures.append(
                f"T={t} full block vs exact chain cosine {cos_blk_exact} below 0.99999")

        accuracy = {
            "router": {"ids_differing": d_ids, "values_differing": d_vals,
                       "weights_differing": d_w, "numel": n_w},
            "routed": {"after_vs_exact_differing": d_routed, "numel": n_routed,
                       "swiglu_differing": d_swiglu, "down_epilogue_differing": d_epi,
                       "before_vs_exact_differing": d_rb, "before_vs_exact_cosine": cos_rb},
            "shared": {"after_vs_before_rel": rel_sh_b, "after_vs_before_cosine": cos_sh_b,
                       "after_vs_cpu_rel": rel_sh_cpu, "after_vs_cpu_cosine": cos_sh_cpu},
            "block": {"after_vs_exact_cosine": cos_blk_exact, "after_vs_exact_rel": rel_blk_exact,
                      "before_vs_exact_cosine": cos_b_exact,
                      "after_vs_before_cosine": cos_blk_b, "after_vs_before_rel": rel_blk_b,
                      "after_vs_cpu_cosine": cos_blk_cpu, "after_vs_cpu_rel": rel_blk_cpu,
                      "before_vs_cpu_cosine": cos_b_cpu,
                      "components_differing": d_sum,
                      "before_vs_served_differing": d_served},
        }

        if args.check_only:
            results["tokens"][str(t)] = {"accuracy": accuracy}
            continue

        # ---------------- timing ----------------------------------------
        samples, warmup, reps = args.samples, args.warmup, args.reps
        rows: dict[str, float] = {}
        raw_samples: dict[str, list] = {}

        def T(name, fn):
            raw_samples[name] = []
            for _ in range(3):
                time_sync(fn, warmup, samples, raw_samples[name])
            ms = _median(raw_samples[name])
            rows[name] = ms
            print(f"  {name:26s} {ms:9.4f} ms", flush=True)
            return ms

        T("router.before", lambda: route_before(block, h))
        T("router.after", lambda: route_topk(h, block.gate.weight, block.top_k))
        T("shared.before", lambda: block.shared_expert(h))
        T("shared.after", lambda: shared_after(block, h))
        T("routed.before", lambda: routed_before(block, h, ids_b, w_b))
        T("routed.h10_only", lambda: _h10_routed(block, h, ids_a, w_a))
        T("routed.after", lambda: routed_after(block, h, ids_a, w_a))
        T("block.before", lambda: block_before(block, h))
        T("block.before_served", lambda: block_before_served(block, h))
        T("block.h10_only", lambda: _h10_block(block, h))
        T("block.after", lambda: block_after(block, h))

        batched: dict[str, float] = {}
        if not args.skip_batched:
            def B(name, fn):
                raw_samples[name + ".batched"] = []
                for _ in range(2):
                    time_batched(fn, reps, 3, raw_samples[name + ".batched"])
                ms = _median(raw_samples[name + ".batched"])
                batched[name] = ms
                print(f"  {name + '.batched':26s} {ms:9.4f} ms", flush=True)
                return ms

            B("router.before", lambda: route_before(block, h))
            B("router.after", lambda: route_topk(h, block.gate.weight, block.top_k))
            B("shared.before", lambda: block.shared_expert(h))
            B("shared.after", lambda: shared_after(block, h))
            B("routed.before", lambda: routed_before(block, h, ids_b, w_b))
            B("routed.h10_only", lambda: _h10_routed(block, h, ids_a, w_a))
            B("routed.after", lambda: routed_after(block, h, ids_a, w_a))
            B("block.before", lambda: block_before(block, h))
            B("block.before_served", lambda: block_before_served(block, h))
            B("block.after", lambda: block_after(block, h))
            B("block.h10_only", lambda: _h10_block(block, h))

        if _is_lazy():
            # One graph per "step" over 8 MoE blocks: distinct inputs avoid
            # CSE, one mark_step at the end (production marks per step, not
            # per layer).  Per-block ms = graph build + execute / 8.
            h_chain = [h + (i * 2e-3) for i in range(8)]
            for name, fn in (("block.before", lambda x: block_before(block, x)),
                             ("block.before_served", lambda x: block_before_served(block, x)),
                             ("block.after", lambda x: block_after(block, x))):
                key = name + ".chain8_per_block"
                raw_samples[key] = []
                for _ in range(2):
                    time_chained(fn, h_chain, 3, raw_samples[key])
                ms = _median(raw_samples[key])
                rows[key] = ms
                print(f"  {key:26s} {ms:9.4f} ms", flush=True)

        speedup = {}
        for comp in ("router", "shared", "routed", "block"):
            b, a = f"{comp}.before", f"{comp}.after"
            if b in rows and a in rows:
                speedup[comp] = rows[b] / rows[a]
                print(f"  x48 {comp:6s} before {48 * rows[b]:8.2f} ms  after {48 * rows[a]:8.2f} ms  "
                      f"speedup {rows[b] / rows[a]:.2f}x")
            bb, aa = f"{comp}.before.batched", f"{comp}.after.batched"
            if bb in batched and aa in batched:
                speedup[comp + ".batched"] = batched[bb] / batched[aa]
                print(f"  x48 {comp + ' (batched)':6s} before {48 * batched[bb]:8.2f} ms  "
                      f"after {48 * batched[aa]:8.2f} ms  speedup {batched[bb] / batched[aa]:.2f}x")

        results["tokens"][str(t)] = {
            "accuracy": accuracy,
            "timing_sync_ms": rows,
            "timing_batched_ms": batched,
            "timing_samples_ms": raw_samples,
            "speedup": speedup,
        }

    results["backends"] = dict(block.experts.backends)
    results["failures"] = failures
    print("\nFAILURES:", failures if failures else "none")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 1 if failures else 0


def _h10_routed(block, h, ids, w):
    with h15_off():
        return grouped_gemv(h, ids, w, block.experts)


def _h10_block(block, h):
    with h15_off():
        return block(h)


if __name__ == "__main__":
    raise SystemExit(main())
