# SPDX-License-Identifier: Apache-2.0
"""H11 Phase 1: one-layer E2E component timer, real weights, Gaudi2 HPU.

Times, per component and with ``torch.hpu.synchronize()`` at every component
boundary (attribution numbers; their sum is therefore >= the real layer):

  hc_attn_read -> mixer (GDN layer 0 / QSA layer 3) -> hc_attn_write ->
  hc_mlp_read -> router_topk -> routed_moe (packed NVFP4; H18 count-ordered
  chunks in prefill, H8/H9/H10 grouped slots in decode) ->
  shared_expert -> hc_mlp_write, plus layer-1 PLE on its own.

HC mapping in this port (``hyperconnection_amd.GatedResidual`` has no
forward_read/forward_write names; those are the STRATA names):
  read  = ``GatedResidual.mix``            (norm -> down+inject -> silu -> up -> gated mean)
  write = ``GatedResidual.combine``        (residual + block * 2*sigmoid(inj/hc))
Production fuses write+next-read into ``combine_and_mix``; the separate
read/write timings here split that fused node, so the HC total is an upper
attribution, not the fused cost. The unsynchronized full-layer numbers below
use the production fused form (``combine_and_mix``), exactly as
``Qwen4ExpDecoderLayer.forward`` does.

Also reported, per regime: the unsynchronized full-layer wall clock (no
synchronize inside the timed loop, results fed into a live accumulator so
nothing can be dead-code eliminated; one synchronize after the loop), and the
48-layer projection 36 x layer0(GDN) + 12 x layer3(QSA) + 1 x PLE(layer 1)
with a subsystem breakdown. PLE lives only on layer 1 (``ple_layer_ids=[2]``,
i.e. ``layer_idx + 1 == 2``); there are 36 GDN and 12 QSA layers in 48.

Regimes: decode T=1 and T=4 (T sequences x 1 token, recurrent/KV state primed
by a real prefill chunk first) and prefill T=512 (one request).

Real weights only, from ~/models/qwen4exp-flashnext-plain:
layer 0 (GDN+HC+MoE), layer 3 (QSA+HC+MoE), layer 1 PLE (kv/norm/conv on HPU,
128 host mmap PLE shards, pin_memory=False -- the production residency).

Note on file names in the original brief: there is no ``gdn.py`` / ``ple.py``
in this tree. GDN is ``vllm_gaudi.models.qwen3_5.HPUGatedDeltaNetAttention``
(the class the decoder constructs via the monkey-patch) and PLE is
``ple_layer.Qwen4ExpPLELayer`` + ``ple_runtime``.

Run:
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_layer_e2e.py
  PT_HPU_LAZY_MODE=1 .venv/bin/python tools/qwen4exp_flashnext/bench_layer_e2e.py

Lazy mode (PT_HPU_LAZY_MODE=1): ``mark_step()`` is issued at every timed
step boundary (after each component / full-layer rep / chained step,
before the synchronize) so each lazy graph executes per step instead of
accumulating across the timing loop. Decode QSA runs at a steady state:
the software KV cache is rolled back to the primed length after every
benchmarked token, so S_kv is 257 on every step (and, with the static
bucket in qsa.py, the attention shape is the 512-row backing capacity)
instead of growing 257, 258, ... and recompiling per token.

Also reported: ``chained_48l_step`` -- 36x layer-0 GDN full layers +
12x layer-3 QSA full layers + 1x layer-1 PLE executed back-to-back
(layer pattern GDN,GDN,GDN,QSA x12, PLE inserted once at the layer-1
position) with a single synchronize at the end (in lazy mode, a
mark_step after each layer so each layer's graph executes back-to-back
instead of accumulating into one uncompilable 48-layer graph): the
true 48-layer step wall clock, as opposed to the per-layer unsync
projection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

# Shared helper, before torch / habana_frameworks (or anything importing
# them): prepend the Qwen4Exp TPC library to GC_KERNEL_PATH if absent,
# preserving existing entries (the host shell points it at the stock TPC
# library, which does not carry the custom NVFP4 GUIDs). Loaded by file
# path so no package __init__ (hence no torch/vLLM import) runs first;
# the same helper also runs from vllm_gaudi.models.qwen4exp at import.
import importlib.util as _ilu  # noqa: E402

_gc_spec = _ilu.spec_from_file_location(
    "_qwen4exp_gc_kernel_path",
    REPO / "vllm-gaudi/vllm_gaudi/models/qwen4exp/_gc_kernel_path.py")
_gc_mod = _ilu.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(_gc_mod)
_gc_mod.ensure_gc_kernel_path()

import torch  # noqa: E402

import habana_frameworks.torch  # noqa: E402,F401  (HPU init; GC_KERNEL_PATH already set above)

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
OUT_DEFAULT = Path(__file__).resolve().parent / "bench_h11_baseline.json"
HIDDEN = 2560
HC_COUNT = 4
TOP_K = 10
NUM_EXPERTS = 512
PRIME_TOKENS = 256  # per-sequence recurrent/KV priming context for decode

_DIST_READY = False
_VLLM_CONFIG = None
_KEEP_CTX = None
_HANDLES: dict = {}
_WEIGHT_MAP: dict | None = None


# --------------------------------------------------------------------------
# checkpoint loading (weight_map + per-shard safe_open handles, as in
# test_moe_block_hpu.py)
# --------------------------------------------------------------------------
def _weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        _WEIGHT_MAP = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    return _WEIGHT_MAP


def _load(name: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = CHECKPOINT / "tensors" / _weight_map()[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(shard, framework="pt", device="cpu")
    return _HANDLES[shard].get_tensor(name)


def _dequant_mxfp8(base: str) -> torch.Tensor:
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    out = dequant_mxfp8(_load(f"{base}.weight"), _load(f"{base}.weight_scale"))
    assert out.shape == _load(f"{base}.weight").shape, base
    return out


# --------------------------------------------------------------------------
# vLLM scaffolding (dummy llama config + real Qwen4Exp text config, as in the
# layer bring-up tests)
# --------------------------------------------------------------------------
def _ensure_dist():
    global _DIST_READY, _VLLM_CONFIG, _KEEP_CTX
    if _DIST_READY:
        return _VLLM_CONFIG
    import vllm_gaudi  # noqa: F401  (registers the HPU platform first)

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
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
    vllm_config.cache_config.enable_prefix_caching = False
    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
        local_rank=0, backend="gloo",
    )
    _KEEP_CTX = set_current_vllm_config(vllm_config)
    _KEEP_CTX.__enter__()  # module-global: GC'ing it resets the global config
    initialize_model_parallel(1, 1)
    _DIST_READY = True
    _VLLM_CONFIG = vllm_config
    return vllm_config


def _cfg():
    return _ensure_dist().model_config.hf_text_config


# --------------------------------------------------------------------------
# module builders (real checkpoint weights)
# --------------------------------------------------------------------------
def build_hc(layer_idx: int, role: str):
    """GatedResidual for layers.<idx>.<role>_hyper_connection."""
    from vllm_gaudi.models.qwen4exp.hyperconnection_amd import GatedResidual
    from vllm_gaudi.models.qwen4exp.hyperconnection_base import HyperConnectionConfig

    cfg = _cfg()
    vllm_config = _ensure_dist()
    _ = vllm_config
    hc = GatedResidual(
        HyperConnectionConfig(
            hc_count=cfg.hc_count, hidden_size=cfg.hidden_size,
            params_dtype=torch.bfloat16, hc_lowrank=cfg.hc_lowrank,
            rms_norm_eps=cfg.rms_norm_eps, hc_per_branch_norm=True,
        ),
        prefix=f"model.layers.{layer_idx}.{role}_hyper_connection",
    )
    base = f"model.language_model.layers.{layer_idx}.{role}_hyper_connection"
    down = _load(f"{base}.input_mix_weight_down.weight").contiguous()
    inject = _load(f"{base}.block_inject_weight.weight").contiguous()
    pad = torch.zeros(hc.pad_size, down.shape[1], dtype=down.dtype)
    merged = torch.cat((down, inject, pad), dim=0).contiguous()
    state = {
        "hc_norm.weight": _load(f"{base}.hc_norm.weight").contiguous(),
        "input_mix_weight_down_block_inject.weight": merged,
        "input_mix_weight_up.weight": _load(f"{base}.input_mix_weight_up.weight").contiguous(),
    }
    params = dict(hc.named_parameters())
    cast = {n: w.to(dtype=params[n].dtype) for n, w in state.items()}
    missing, unexpected = hc.load_state_dict(cast, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return hc.to("hpu")


def build_gdn():
    """Layer-0 HPUGatedDeltaNetAttention (test_gdn_layer_hpu.py recipe)."""
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

    cfg = _cfg()
    vllm_config = _ensure_dist()
    layer = "model.language_model.layers.0.linear_attn"
    qkv = _dequant_mxfp8(f"{layer}.in_proj_qkv")
    z = _dequant_mxfp8(f"{layer}.in_proj_z")
    b = _dequant_mxfp8(f"{layer}.in_proj_b")
    a = _dequant_mxfp8(f"{layer}.in_proj_a")
    state = {
        "in_proj_qkvz.weight": torch.cat((qkv, z), dim=0).contiguous(),
        "in_proj_ba.weight": torch.cat((b, a), dim=0).contiguous(),
        "conv1d.weight": _load(f"{layer}.conv1d.weight").contiguous(),
        "A_log": _load(f"{layer}.A_log"),
        "dt_bias": _load(f"{layer}.dt_bias"),
        "norm.weight": _load(f"{layer}.norm.weight"),
        "out_proj.weight": _dequant_mxfp8(f"{layer}.out_proj"),
    }
    torch.manual_seed(0)
    mod = HPUGatedDeltaNetAttention(
        cfg, vllm_config=vllm_config, prefix="model.layers.0.linear_attn",
        gqa_interleaved_layout=False,
    )
    params = dict(mod.named_parameters())
    cast = {n: w.to(dtype=params[n].dtype) if n in params else w for n, w in state.items()}
    missing, unexpected = mod.load_state_dict(cast, strict=False)
    assert not [m for m in missing if not m.endswith(".bias")], missing
    assert not unexpected, unexpected
    mod.to(dtype=torch.bfloat16)
    return mod.to("hpu")


def build_qsa():
    """Layer-3 Qwen4ExpQSAAttention (test_qsa_layer_hpu.py recipe)."""
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention

    vllm_config = _ensure_dist()
    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig(
        hidden_size=int(text["hidden_size"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        max_position_embeddings=int(text["max_position_embeddings"]),
        rms_norm_eps=float(text["rms_norm_eps"]),
        indexer_budget=int(text["indexer_budget"]),
        indexer_compress_ratio=int(text["indexer_compress_ratio"]),
        indexer_n_heads=int(text["indexer_n_heads"]),
        indexer_kv_heads=int(text["indexer_kv_heads"]),
        indexer_head_dim=int(text["indexer_head_dim"]),
        extras={"rope_parameters": text["rope_parameters"]},
    )
    layer = "model.language_model.layers.3.self_attn"
    state = {
        "qkv_proj.weight": torch.cat((
            _dequant_mxfp8(f"{layer}.q_proj"),
            _dequant_mxfp8(f"{layer}.k_proj"),
            _dequant_mxfp8(f"{layer}.v_proj"),
        ), dim=0).contiguous(),
        "q_norm.weight": _load(f"{layer}.q_norm.weight"),
        "k_norm.weight": _load(f"{layer}.k_norm.weight"),
        "o_proj.weight": _dequant_mxfp8(f"{layer}.o_proj"),
        "indexer.index_qk_proj.weight": _dequant_mxfp8(f"{layer}.indexer.index_qk_proj"),
        "indexer.q_layernorm.weight": _load(f"{layer}.indexer.q_layernorm.weight"),
        "indexer.k_layernorm.weight": _load(f"{layer}.indexer.k_layernorm.weight"),
    }
    torch.manual_seed(0)
    mod = Qwen4ExpQSAAttention(
        vllm_config=vllm_config, config=cfg, layer_id=3, prefix="model.layers.3.self_attn",
    )
    missing, unexpected = mod.load_state_dict(state, strict=False)
    assert not [m for m in missing if "rotary" not in m and "attn." not in m], missing
    assert not unexpected, unexpected
    mod.to(torch.bfloat16)
    return mod.to("hpu")


def build_moe(layer_idx: int):
    """Qwen4ExpSparseMoeBlock for one layer (test_moe_block_hpu.py recipe)."""
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpSparseMoeBlock
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    cfg = _cfg()
    vllm_config = _ensure_dist()
    if not hasattr(torch.ops.custom_op, "custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2"):
        ext = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob(
            "hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
        torch.ops.load_library(str(ext))
    block = Qwen4ExpSparseMoeBlock(vllm_config, prefix=f"model.layers.{layer_idx}.mlp")
    block.load_state_dict({}, strict=False)

    base = f"model.language_model.layers.{layer_idx}.mlp"
    weights: list[tuple[str, torch.Tensor]] = [
        ("gate.weight", _load(f"{base}.gate.weight").contiguous()),
        ("shared_expert_gate.weight", _load(f"{base}.shared_expert_gate.weight").contiguous()),
        ("shared_expert.down_proj.weight", _load(f"{base}.shared_expert.down_proj.weight").contiguous()),
    ]
    for shard_id, kind in enumerate(("gate_proj", "up_proj")):
        w = dequant_mxfp8(
            _load(f"{base}.shared_expert.{kind}.weight"),
            _load(f"{base}.shared_expert.{kind}.weight_scale"))
        w.shard_id = shard_id  # type: ignore[attr-defined]
        weights.append(("shared_expert.gate_up_proj.weight", w))
    for expert in range(cfg.num_experts):
        for kind in ("gate_proj", "up_proj", "down_proj"):
            eb = f"{base}.experts.{expert}.{kind}.weight"
            weights.append((f"experts.{expert}.{kind}.weight", _load(eb).contiguous()))
            weights.append((f"experts.{expert}.{kind}.weight_scale",
                            _load(eb + "_scale").view(torch.uint8)))
            weights.append((f"experts.{expert}.{kind}.weight_scale_2", _load(eb + "_scale_2")))
    block.load_weights(iter(weights))
    assert block.experts.gate_proj_packed.shape[0] == cfg.num_experts
    assert block.experts.gate_proj_scale2.abs().sum() > 0, "scale2 not loaded"
    block.to(dtype=torch.bfloat16)
    return block.to("hpu")


def build_ple():
    """Layer-1 Qwen4ExpPLELayer + host mmap n-gram tables (test_ple recipe,
    but all 128 shards file-backed instead of only the touched ones pinned)."""
    from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer
    from vllm_gaudi.models.qwen4exp.ple_runtime import load_ple_ngram

    cfg = _cfg()
    vllm_config = _ensure_dist()
    layer = Qwen4ExpPLELayer(cfg, vllm_config, layer_idx=1, prefix="model.layers.1.ple")
    base = "model.language_model.layers.1.ple"
    state = {
        "kv_proj.weight": torch.cat((
            _load(f"{base}.key_proj.weight"), _load(f"{base}.value_proj.weight"),
        ), dim=0).contiguous(),
        "norm_key.weight": _load(f"{base}.norm_key.weight"),
        "norm_query.weight": _load(f"{base}.norm_query.weight"),
        "norm_conv.weight": _load(f"{base}.norm_conv.weight"),
        "conv1d.weight": _load(f"{base}.conv1d.weight").contiguous(),
    }
    params = dict(layer.named_parameters())
    cast = {n: w.to(dtype=params[n].dtype) for n, w in state.items()}
    missing, unexpected = layer.load_state_dict(cast, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    layer.to(dtype=torch.bfloat16)
    layer = layer.to("hpu")
    layer.attach_ngram(load_ple_ngram(
        CHECKPOINT, layer_idx=1, heads_per_ngram=cfg.heads_per_ngram,
        ngram_size=cfg.ngram_size, pin_memory=False, mmap=True,
    ))
    return layer


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------
def _is_lazy() -> bool:
    return os.environ.get("PT_HPU_LAZY_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _mark_step() -> None:
    """Flush the current lazy graph (no-op in eager mode).

    Must be called at a step boundary *before* ``torch.hpu.synchronize()``
    so the graph built by the step just enqueued is the one that executes
    (and is timed), instead of accumulating into the next step's graph.
    """
    if not _is_lazy():
        return
    try:
        import habana_frameworks.torch.core as _htcore

        _htcore.mark_step()
    except Exception:
        pass


def _sync_step() -> None:
    """Step boundary used outside the timed loops: mark_step + synchronize."""
    _mark_step()
    torch.hpu.synchronize()


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _checksum(out) -> float:
    if out is None:
        return 0.0
    if isinstance(out, torch.Tensor):
        return float(out.float().sum().item())
    if isinstance(out, (tuple, list)):
        return sum(_checksum(o) for o in out)
    return 0.0


def time_sync(fn, warmup: int, samples: int) -> tuple[float, float]:
    """Per-component attribution: synchronize around every single call.

    In lazy mode ``mark_step()`` precedes each synchronize (warmup, timed
    and the initial drain) so every call executes as its own graph.
    """
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
    assert check == check and abs(check) != float("inf"), f"non-finite checksum {check}"
    return _median(times), check


def time_full_layer(fn, reps: int, samples: int) -> tuple[float, float]:
    """Unsynchronized wall clock: no sync between reps (or between the layer's
    components); the output of every rep feeds a live accumulator tensor
    (nothing is dead code), and a single synchronize after the rep loop --
    before the clock stops -- makes the elapsed time include completion of
    the last rep, as in bench_decode_moe._batched.

    In lazy mode ``mark_step()`` is issued after every rep (and after the
    warmup reps): it flushes that rep's graph for execution without a host
    synchronize, so reps remain unsynchronized back-to-back steps exactly
    as in eager, rather than fusing into one ever-growing lazy graph.
    """
    acc = torch.zeros((), dtype=torch.float32, device="hpu")

    def one_rep():
        nonlocal acc
        with torch.no_grad():
            out = fn()
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
        torch.hpu.synchronize()  # once, after the loop; no sync between reps
        times.append((time.perf_counter() - t0) * 1e3 / reps)
    check = float(acc.float().item())
    assert check == check and abs(check) != float("inf"), f"non-finite full-layer checksum {check}"
    return _median(times), check


# --------------------------------------------------------------------------
# GDN metadata / state handling
# --------------------------------------------------------------------------
def _gdn_metadata(num_tokens, num_seqs, is_prompt, device="hpu", chunk=None, has_initial=False):
    idx = torch.arange(num_seqs, dtype=torch.int32, device=device)
    if is_prompt and num_seqs > 1 and chunk:
        qsl = torch.tensor([i * chunk for i in range(num_seqs + 1)], dtype=torch.int32, device=device)
    elif is_prompt:
        qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    else:
        qsl = torch.tensor(list(range(num_seqs + 1)), dtype=torch.int32, device=device)
    return SimpleNamespace(
        is_prompt=is_prompt,
        load_indices_tensor=idx,
        store_indices_tensor=idx,
        query_start_loc_p=qsl,
        has_initial_states_p=torch.tensor([has_initial] * num_seqs, device=device),
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        gdn_initial_state_indices=None,
        dflash_full_query=False,
    )


def _gdn_bind(gdn, num_slots):
    conv_shape, ssm_shape = gdn.get_state_shape()
    conv_dtype, ssm_dtype = gdn.get_state_dtype()
    gdn.kv_cache = (
        torch.zeros((num_slots, *conv_shape), dtype=conv_dtype, device="hpu"),
        torch.zeros((num_slots, *ssm_shape), dtype=ssm_dtype, device="hpu"),
    )


def _gdn_run(gdn, hidden, meta):
    from vllm.forward_context import set_forward_context

    with set_forward_context(meta, _VLLM_CONFIG, num_tokens=hidden.shape[0]):
        return gdn(hidden)


# --------------------------------------------------------------------------
# main bench
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--decode-warmup", type=int, default=3)
    ap.add_argument("--decode-samples", type=int, default=7)
    ap.add_argument("--prefill-warmup", type=int, default=2)
    ap.add_argument("--prefill-samples", type=int, default=5)
    ap.add_argument("--regimes", default="",
                    help="comma-separated subset of decode_T1,decode_T4,prefill (default: all)")
    args = ap.parse_args()

    assert KERNEL.is_file(), KERNEL
    print(f"bench_layer_e2e: GC_KERNEL_PATH={os.environ['GC_KERNEL_PATH']}")
    print(f"  checkpoint={CHECKPOINT}  lazy={os.environ.get('PT_HPU_LAZY_MODE')}")

    _ensure_dist()
    cfg = _cfg()
    gen = torch.Generator().manual_seed(2026)

    def randn(*shape, scale=0.1):
        return (torch.randn(*shape, generator=gen) * scale).bfloat16()

    print("building layer 0 (GDN + HC + MoE) ...", flush=True)
    hc0_attn, hc0_mlp = build_hc(0, "attn"), build_hc(0, "mlp")
    gdn = build_gdn()
    moe0 = build_moe(0)
    print("building layer 3 (QSA + HC + MoE) ...", flush=True)
    hc3_attn, hc3_mlp = build_hc(3, "attn"), build_hc(3, "mlp")
    qsa = build_qsa()
    moe3 = build_moe(3)
    print("building layer 1 PLE (128 host mmap shards) ...", flush=True)
    ple = build_ple()
    _sync_step()
    print("all modules on HPU with real weights", flush=True)

    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts_grouped,
        compact_expert_ids,
        grouped_from_ids,
        prefill_grouped_enabled,
        prefill_grouped_h18,
        route_topk,
        shared_expert_forward_h15,
    )

    def run_shared_expert(moe_block, x):
        """Exact production shared-expert path (model.py SparseMoeBlock).

        Qwen4ExpSparseMoeBlock.forward does::

            shared_out = shared_expert_forward_h15(self.shared_expert, hidden)
            out = out + (shared_out if shared_out is not None
                         else self.shared_expert(hidden))

        The isolated timer must time that helper (merged gate_up+gate
        GEMM -> fused SwiGLU/sigmoid epilogue -> down GEMM), not a direct
        ``moe.shared_expert(x)`` call, which skips the H15 fused path
        entirely. Only the ``+ routed_out`` add belongs to the full MoE,
        so the component returns just the shared contribution.
        """
        shared_out = shared_expert_forward_h15(moe_block.shared_expert, x)
        return shared_out if shared_out is not None else moe_block.shared_expert(x)

    results: dict = {
        "benchmark": "bench_layer_e2e (H11 Phase 1)",
        "checkpoint": str(CHECKPOINT),
        "kernel_lib": os.environ["GC_KERNEL_PATH"],
        "pt_hpu_lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "mode_notes": [
            "component ms: torch.hpu.synchronize() around every call, median (attribution; sum >= real layer)",
            "full_layer ms: no synchronize inside the timed loop, live accumulator, one sync after the loop",
            "HC read=GatedResidual.mix, write=GatedResidual.combine; production fuses write+next read (combine_and_mix), used by the full-layer path",
            "no gdn.py/ple.py exists in this tree: GDN=HPUGatedDeltaNetAttention (qwen3_5), PLE=ple_layer+ple_runtime",
            "PLE tables: 128 host mmap shards, pin_memory=False; kv/norm/conv weights on HPU",
            "shared_expert component: production shared_expert_forward_h15 (packed_moe) "
            "with stock-MLP fallback, exactly as Qwen4ExpSparseMoeBlock.forward",
            "routed_moe component (prefill): production H18 prefill_grouped_h18 first, "
            "falling back to compact_expert_ids + dequant + apply_routed_experts_grouped "
            "only when it returns None (switch off / non-HPU / slots < "
            "PREFILL_H18_MIN_SLOTS / empty plan), exactly as Qwen4ExpSparseMoeBlock.forward",
            "lazy mode: mark_step() at every step boundary before synchronize; decode QSA "
            "cache rolled back to primed length after each token (steady-state S_kv) "
            "+ static-capacity attention in qsa.py",
            "chained_48l_step: 36x GDN + 12x QSA full layers + 1x PLE back-to-back, one step boundary/sync at the end",
        ],
        "geometry": {
            "hidden": cfg.hidden_size, "hc_count": cfg.hc_count, "hc_lowrank": cfg.hc_lowrank,
            "experts": cfg.num_experts, "top_k": cfg.num_experts_per_tok,
            "moe_intermediate": cfg.moe_intermediate_size,
            "layers_total": 48, "gdn_layers": 36, "qsa_layers": 12, "ple_layers": 1,
            "ple_layer_idx": 1,
        },
        "regimes": {},
    }

    regimes = [
        ("decode_T1", "decode", 1),
        ("decode_T4", "decode", 4),
        (f"prefill_T{args.prefill_tokens}", "prefill", args.prefill_tokens),
    ]
    if args.regimes:
        wanted = {r.strip() for r in args.regimes.split(",") if r.strip()}
        regimes = [r for r in regimes if r[0] in wanted or r[1] in wanted]
        assert regimes, f"no regime matches --regimes {args.regimes!r}"

    for regime_name, mode, tokens in regimes:
        is_prefill = mode == "prefill"
        warmup = args.prefill_warmup if is_prefill else args.decode_warmup
        samples = args.prefill_samples if is_prefill else args.decode_samples
        # Fewer samples for the heavy prefill components/full layer.
        pf_samples = 3 if is_prefill else samples
        print(f"\n===== {regime_name} (mode={mode}, T={tokens}) =====", flush=True)
        reg: dict = {"mode": mode, "tokens": tokens, "layers": {}, "checksums": {}}

        hidden_multi = randn(tokens, HC_COUNT * HIDDEN, scale=0.2).to("hpu")

        # ---- mixer closures with realistic state --------------------------
        # GDN
        if is_prefill:
            _gdn_bind(gdn, 2)
            gdn_meta = _gdn_metadata(tokens, 1, True)
        else:
            _gdn_bind(gdn, tokens + 2)
            prime = randn(tokens * PRIME_TOKENS, HIDDEN, scale=0.1).to("hpu")
            _gdn_run(gdn, prime, _gdn_metadata(tokens * PRIME_TOKENS, tokens, True,
                                               chunk=PRIME_TOKENS, has_initial=False))
            _sync_step()
            gdn_meta = _gdn_metadata(tokens, tokens, False, has_initial=True)
        gdn_calls = {"n": 0}

        def run_gdn(block_input):
            gdn_calls["n"] += 1
            return _gdn_run(gdn, block_input, gdn_meta)

        # QSA
        qsa_state = {"next_pos": PRIME_TOKENS}
        if is_prefill:
            def run_qsa(block_input):
                qsa.reset_bringup_cache()
                pos = torch.arange(tokens, device="hpu")
                qsl = torch.tensor([0, tokens], dtype=torch.int32, device="hpu")
                return qsa(pos, block_input, query_start_loc=qsl)
        else:
            qsa.reset_bringup_cache()
            prime_in = randn(tokens * PRIME_TOKENS, HIDDEN, scale=0.1).to("hpu")
            prime_pos = torch.cat([torch.arange(PRIME_TOKENS, device="hpu")] * tokens)
            prime_qsl = torch.tensor([i * PRIME_TOKENS for i in range(tokens + 1)],
                                     dtype=torch.int32, device="hpu")
            qsa(prime_pos, prime_in, query_start_loc=prime_qsl)
            _sync_step()

            def _rollback_qsa():
                # Steady-state decode: restore every slot to the primed
                # length so the next benchmarked token appends at the
                # same position / S_kv (257) instead of growing the cache
                # 257, 258, ... -- in lazy mode that growth is a new
                # attention shape (recipe) per token. Truncation is
                # length-only on the slot's own backing buffers, so it
                # does not disturb the graph just enqueued; the token
                # written by that graph is simply overwritten next step.
                for state in qsa._bringup:
                    if state is None:
                        continue
                    state.truncate_tokens(PRIME_TOKENS)
                    state.truncate_pooled(PRIME_TOKENS // qsa.indexer.compress_ratio)
                qsa_state["next_pos"] = PRIME_TOKENS

            def run_qsa(block_input):
                pos = torch.full((tokens,), qsa_state["next_pos"], dtype=torch.int64, device="hpu")
                qsa_state["next_pos"] += 1
                qsl = torch.tensor(list(range(tokens + 1)), dtype=torch.int32, device="hpu")
                try:
                    return qsa(pos, block_input, query_start_loc=qsl)
                finally:
                    _rollback_qsa()

        layer_defs = [
            ("layer0_gdn", hc0_attn, hc0_mlp, run_gdn, moe0),
            ("layer3_qsa", hc3_attn, hc3_mlp, run_qsa, moe3),
        ]
        full_layer_fns: dict = {}

        for layer_name, hc_attn, hc_mlp, run_mixer, moe in layer_defs:
            print(f"-- {layer_name} --", flush=True)
            # Real chain once (untimed) to obtain the tensors each stage consumes.
            with torch.no_grad():
                _, block_attn, inj_attn = hc_attn.mix(hidden_multi)
                attn_out = run_mixer(block_attn)
                hidden_mid, block_mlp, inj_mlp = hc_mlp.combine_and_mix(hidden_multi, attn_out, inj_attn)
                topk_ids, topk_w = route_topk(block_mlp, moe.gate.weight, TOP_K)
                mlp_out = moe(block_mlp)
            _sync_step()

            def routed(block_input, _moe=moe, _ids=topk_ids, _w=topk_w):
                if is_prefill:
                    # Exactly Qwen4ExpSparseMoeBlock.forward's prefill branch:
                    # H18 first (count-ordered expert chunks, one host
                    # transfer for the whole shape plan, per-chunk cap), and
                    # only when H18 declines (``None``: switch off, non-HPU,
                    # dtype/shape guard, slots < PREFILL_H18_MIN_SLOTS,
                    # empty plan) the H9(a) compact-ids + dequant +
                    # expert-major grouped buckets path runs unchanged.
                    out = None
                    if prefill_grouped_enabled():
                        out = prefill_grouped_h18(
                            block_input, _ids, _w, _moe.experts)
                    if out is not None:
                        return out
                    unique, local = compact_expert_ids(_ids, NUM_EXPERTS)
                    gate_w = _moe.experts.dequant("gate_proj", unique)
                    up_w = _moe.experts.dequant("up_proj", unique)
                    down_w = _moe.experts.dequant("down_proj", unique)
                    if _is_lazy():
                        # Drain the dequant graphs before the grouped
                        # bucket build: its host cap read segments the
                        # lazy graph, and the dequant outputs (not
                        # ancestors of the read) must be real tensors
                        # by then, not pending graph outputs.
                        _sync_step()
                    return apply_routed_experts_grouped(
                        block_input, local, _w, gate_w, up_w, down_w)
                return grouped_from_ids(block_input, _ids, _w, _moe.experts)

            comp_fns = [
                ("hc_attn_read", lambda: hc_attn.mix(hidden_multi)[1]),
                ("mixer", lambda: run_mixer(block_attn)),
                ("hc_attn_write", lambda: hc_mlp.combine(hidden_multi, attn_out, inj_attn)),
                ("hc_mlp_read", lambda: hc_mlp.mix(hidden_mid)[1]),
                ("router_topk", lambda: route_topk(block_mlp, moe.gate.weight, TOP_K)[0]),
                ("routed_moe", lambda: routed(block_mlp)),
                ("shared_expert", lambda _moe=moe: run_shared_expert(_moe, block_mlp)),
                ("hc_mlp_write", lambda: hc_mlp.combine(hidden_mid, mlp_out, inj_mlp)),
            ]
            comps: dict[str, float] = {}
            for comp_name, fn in comp_fns:
                heavy = is_prefill and comp_name in ("mixer", "routed_moe", "shared_expert")
                ms, check = time_sync(fn, warmup=1 if heavy else warmup,
                                      samples=pf_samples if heavy else samples)
                comps[comp_name] = ms
                reg["checksums"][f"{layer_name}.{comp_name}"] = check
                print(f"  {comp_name:16s} {ms:9.4f} ms", flush=True)

            # Unsynchronized full layer (production fused HC form).
            # attn_out/inj_attn are bound as defaults so a stored closure
            # (chained_48l_step below) still uses this layer's tensors
            # after the loop has moved on to the next layer.
            def full_layer(_hc_attn=hc_attn, _hc_mlp=hc_mlp, _mixer=run_mixer, _moe=moe,
                           _attn_out=attn_out, _inj_attn=inj_attn):
                h, blk, inj = _hc_attn.combine_and_mix(hidden_multi, _attn_out, _inj_attn)
                out_attn = _mixer(blk)
                h, blk2, inj2 = _hc_mlp.combine_and_mix(h, out_attn, inj)
                return _moe(blk2)

            full_layer_fns[layer_name] = full_layer
            full_ms, full_check = time_full_layer(
                full_layer, reps=1 if is_prefill else 10, samples=pf_samples)
            reg["checksums"][f"{layer_name}.full_layer"] = full_check
            comp_sum = sum(comps.values())
            reg["layers"][layer_name] = {
                "components_ms": comps,
                "components_sum_ms": comp_sum,
                "full_layer_unsync_ms": full_ms,
            }
            print(f"  {'SUM(sync comps)':16s} {comp_sum:9.4f} ms   full_layer_unsync {full_ms:9.4f} ms",
                  flush=True)

        # ---- PLE (layer 1 only) -------------------------------------------
        ple_ids_cpu = torch.randint(100, 1000, (tokens,), generator=gen, dtype=torch.int32)
        if is_prefill:
            ple_qsl = torch.tensor([0, tokens], dtype=torch.int32)
            ple_ctx = torch.full((1, 2), ple.eos_token_id, dtype=torch.int32)
            ple_pos = torch.arange(tokens, dtype=torch.int64)
        else:
            ple_qsl = torch.tensor(list(range(tokens + 1)), dtype=torch.int32)
            ple_ctx = torch.full((tokens, 2), ple.eos_token_id, dtype=torch.int32)
            ple_pos = torch.full((tokens,), PRIME_TOKENS, dtype=torch.int64)
        ple._bringup_conv_state = None

        def run_ple():
            return ple(hidden_multi, ple_ids_cpu.to("hpu"), ple_qsl.to("hpu"),
                       ple_ctx.to("hpu"), positions=ple_pos.to("hpu"))

        ple_ms, ple_check = time_sync(run_ple, warmup=1 if is_prefill else warmup,
                                      samples=pf_samples if is_prefill else samples)
        reg["ple_ms"] = ple_ms
        reg["checksums"]["ple"] = ple_check
        print(f"  {'ple (layer1)':16s} {ple_ms:9.4f} ms", flush=True)

        # ---- chained 48-layer step --------------------------------------
        # True back-to-back step: 36x layer-0 GDN + 12x layer-3 QSA full
        # layers in the model's GDN,GDN,GDN,QSA x12 pattern, plus the
        # single layer-1 PLE inserted after the first GDN block, all
        # enqueued before any synchronize (one synchronize, in
        # time_chained below, after the whole chain) -- unlike the
        # projection below, which multiplies individually-timed layers
        # and never pays the real inter-layer launch/queue behaviour of
        # one 48-layer step. Each layer output is folded into a live
        # scalar sink so nothing can be dead-code eliminated and lazy
        # intermediates stay consumed.
        #
        # Lazy mode: mark_step() after *each* layer (inside the chain).
        # A single 48-layer lazy graph does not compile here -- QSA's
        # host-side span read (.tolist() on positions) forces execution
        # of everything accumulated so far mid-chain, and that fused
        # graph fails with synStatus 26. Per-layer mark_step flushes
        # each layer's graph for back-to-back async execution while the
        # host keeps enqueueing; the single synchronize at the end of
        # the step is what makes this a step wall clock, not 48 drains.
        def chained_48l_step(_fns=full_layer_fns, _run_ple=run_ple):
            # Lazy prefill drains per layer: the prefill closures reset
            # state per call (QSA bringup cache, GDN slot, GB-scale
            # dequant buffers), so purely-marked back-to-back graphs
            # reuse storage that pending graphs still read -> NaN.
            # Decode (the lazy production case) keeps mark-only
            # back-to-back execution with the single final sync.
            boundary = _sync_step if (_is_lazy() and is_prefill) else _mark_step
            debug = os.environ.get("BENCH_CHAINED_DEBUG") == "1"

            def _fold(tag, out):
                s = out.float().sum()
                if debug:
                    _sync_step()
                    print(f"    chained {tag}: sum={float(s):.4f} "
                          f"finite={bool(torch.isfinite(s))}", flush=True)
                return s

            sink = hidden_multi.new_zeros((), dtype=torch.float32)
            ple_done = False
            for _repeat in range(12):
                for _gdn_idx in range(3):
                    sink = sink + _fold(
                        f"gdn r{_repeat}x{_gdn_idx}", _fns["layer0_gdn"]())
                    boundary()
                    if not ple_done:
                        sink = sink + _fold("ple", _run_ple())
                        boundary()
                        ple_done = True
                sink = sink + _fold(f"qsa r{_repeat}", _fns["layer3_qsa"]())
                boundary()
            return sink

        # Reset recurrent/conv state before every chained rep (untimed):
        # the chain reuses layer 0's GDN 36x per step, so carrying its
        # state across warmup+sample reps is hundreds of extra decode
        # tokens on one slot and diverges to NaN (the per-layer timers
        # above never accumulate that far). A real 48-layer step has
        # 36 *distinct* GDN states, each advanced one token. QSA needs
        # no reset: run_qsa already rolls back to the primed length.
        def _reset_chained_state():
            _gdn_bind(gdn, 2 if is_prefill else tokens + 2)
            ple._bringup_conv_state = None
            _sync_step()

        def time_chained(fn, reset, warmup: int, samples: int):
            check = 0.0
            for _ in range(warmup):
                reset()
                with torch.no_grad():
                    out = fn()
                _sync_step()
                check += _checksum(out)
            times = []
            for _ in range(samples):
                reset()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = fn()
                _sync_step()  # the one synchronize for the whole step
                times.append((time.perf_counter() - t0) * 1e3)
                check += _checksum(out)
            assert check == check and abs(check) != float("inf"), (
                f"non-finite chained checksum {check}")
            return _median(times), check

        chained_ms, chained_check = time_chained(
            chained_48l_step, _reset_chained_state,
            warmup=2, samples=pf_samples)
        reg["chained_48l_step_ms"] = chained_ms
        reg["checksums"]["chained_48l_step"] = chained_check
        print(f"  {'chained_48l_step':16s} {chained_ms:9.4f} ms "
              f"(36xGDN + 12xQSA + 1xPLE, "
              f"{'per-layer drain, lazy prefill' if _is_lazy() and is_prefill else 'one sync'})",
              flush=True)
        results["regimes"][regime_name] = reg

    # ---- 48-layer projection ----------------------------------------------
    projections = {}
    for regime_name, reg in results["regimes"].items():
        l0 = reg["layers"]["layer0_gdn"]["components_ms"]
        l3 = reg["layers"]["layer3_qsa"]["components_ms"]
        ple_ms = reg["ple_ms"]

        def agg(key, src0=l0, src3=l3):
            return 36.0 * src0[key] + 12.0 * src3[key]

        breakdown = {
            "GDN": 36.0 * l0["mixer"],
            "QSA": 12.0 * l3["mixer"],
            "HC": (36.0 * sum(l0[k] for k in ("hc_attn_read", "hc_attn_write", "hc_mlp_read", "hc_mlp_write"))
                   + 12.0 * sum(l3[k] for k in ("hc_attn_read", "hc_attn_write", "hc_mlp_read", "hc_mlp_write"))),
            "RoutedMoE": agg("routed_moe"),
            "SharedExpert": agg("shared_expert"),
            "Router": agg("router_topk"),
            "PLE": ple_ms,
        }
        total_sync = sum(breakdown.values())
        total_full = (36.0 * reg["layers"]["layer0_gdn"]["full_layer_unsync_ms"]
                      + 12.0 * reg["layers"]["layer3_qsa"]["full_layer_unsync_ms"] + ple_ms)
        proj = {
            "breakdown_ms": breakdown,
            "breakdown_pct": {k: (100.0 * v / total_sync if total_sync else 0.0)
                              for k, v in breakdown.items()},
            "projected_48layer_sync_sum_ms": total_sync,
            "projected_48layer_full_layer_unsync_ms": total_full,
            "chained_48l_step_ms": reg.get("chained_48l_step_ms"),
            "chained_vs_projected_unsync_ratio": (
                reg["chained_48l_step_ms"] / total_full
                if reg.get("chained_48l_step_ms") and total_full else None),
            "tokens": reg["tokens"],
        }
        if reg["mode"] == "decode":
            proj["projected_ms_per_token_sync"] = total_sync / reg["tokens"]
            proj["vs_reference_273ms_per_token_pct"] = (
                100.0 * (total_sync / reg["tokens"]) / 273.0)
        else:
            proj["prefill_tokens_per_s_at_projected_sync"] = (
                reg["tokens"] / (total_sync / 1e3) if total_sync else 0.0)
        projections[regime_name] = proj
    results["projection_48layer"] = projections

    args.json.write_text(json.dumps(results, indent=2) + "\n")

    # ---- printed tables -----------------------------------------------------
    for regime_name, reg in results["regimes"].items():
        print(f"\n### {regime_name}: component breakdown (synchronized median, ms)")
        print(f"{'component':16s} {'layer0_GDN':>11s} {'layer3_QSA':>11s}")
        keys = list(reg["layers"]["layer0_gdn"]["components_ms"].keys())
        for k in keys:
            print(f"{k:16s} {reg['layers']['layer0_gdn']['components_ms'][k]:11.4f} "
                  f"{reg['layers']['layer3_qsa']['components_ms'][k]:11.4f}")
        print(f"{'SUM':16s} {reg['layers']['layer0_gdn']['components_sum_ms']:11.4f} "
              f"{reg['layers']['layer3_qsa']['components_sum_ms']:11.4f}")
        print(f"{'full_unsync':16s} {reg['layers']['layer0_gdn']['full_layer_unsync_ms']:11.4f} "
              f"{reg['layers']['layer3_qsa']['full_layer_unsync_ms']:11.4f}")
        print(f"{'ple':16s} {reg['ple_ms']:11.4f} (x1)")
        print(f"{'chained_48l':16s} {reg['chained_48l_step_ms']:11.4f} (measured chained step)")
        p = projections[regime_name]
        print(f"projected 48-layer (sync sum): {p['projected_48layer_sync_sum_ms']:.3f} ms | "
              f"(full-layer unsync): {p['projected_48layer_full_layer_unsync_ms']:.3f} ms | "
              f"(chained measured): {p['chained_48l_step_ms']:.3f} ms")
        for k, v in p["breakdown_ms"].items():
            print(f"  {k:14s} {v:10.3f} ms  {p['breakdown_pct'][k]:6.2f}%")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
