# SPDX-License-Identifier: Apache-2.0
"""H23 Phase 1: QSA 12-layer acceleration bench + accuracy harness.

Real Layer-3 QSA weights (``model.language_model.layers.3.self_attn``) on the
Gaudi2, both attention classes the model builds
(``Qwen4ExpQSAAttention`` and the production ``QuantQwen4ExpQSAAttention``),
four regimes, and an A/B of the four H23 items in one process (so the HPU
init and the weight dequant are paid once and both sides of a comparison run
on the same card state).

Regimes
-------
``decode_T1``       Decode, batch 1, S_kv = 257 (256-token priming + 1)
``decode_T4``       Decode, 4 sequences x 1 token, S_kv = 257 per slot
``long_decode_T1``  Decode, batch 1, S_kv = 2560 > indexer budget 2048,
                    ``VLLM_HPU_PREFILL_TILE_REPLAY=1`` (the 2107-token serve shape)
``prefill_T512``    Prefill, one 512-token chunk

What is timed, per (config, variant, regime)
--------------------------------------------
``components``      one ``torch.hpu.synchronize()`` per phase (attribution; the
                    sum is an upper bound, and on this card a sync costs
                    ~25-120 us, see ``probe_h23_ops.py`` [6])
``full_layer``      no synchronize inside the loop, live accumulator, one sync
                    after -- the same definition as bench_layer_e2e
``full_layer_cached_pos``  same, but the *same* ``positions`` tensor for every
                    rep: isolates the layer's own cost when the step's span
                    plan already exists (H23-2's cross-layer reuse)
``chained_12l``     12 back-to-back QSA layer calls with ONE ``positions``
                    tensor and one sync at the end: the real 12-layer step,
                    which is where per-layer host syncs show up

Accuracy is measured, not asserted away: every layer output is kept and
compared against the ``off`` (pre-H23) config of the same regime, so the
report can state the exact numeric delta each item introduces.

Env
---
``--configs off,on`` selects env settings applied between measurements:
``off`` = ``VLLM_QWEN4EXP_QSA_H23=0`` (pre-H23), ``on`` = ``=1`` (adopted),
plus per-item configs ``attn,spans,tile,fuse`` (only that item on).
H28 adds ``h28_off`` (H23 on, H28 off = pre-H28 behavior) and the per-item
``h28_sparse``/``h28_rope``, so ``--configs off,h28_off,on`` is the H23+H28
A/B in one process.

Run:
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h23_qsa.py
  PT_HPU_LAZY_MODE=0 .venv/bin/python tools/qwen4exp_flashnext/bench_h23_qsa.py \
      --configs off,on --json bench_h23_after.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
sys.path.insert(0, str(REPO / "vllm-gaudi"))

# Shared helper, before torch / habana_frameworks: prepend the Qwen4Exp TPC
# library to GC_KERNEL_PATH (the HC/GDN/MoE custom GUIDs live there).
import importlib.util as _ilu  # noqa: E402

_gc_spec = _ilu.spec_from_file_location(
    "_qwen4exp_gc_kernel_path",
    REPO / "vllm-gaudi/vllm_gaudi/models/qwen4exp/_gc_kernel_path.py")
_gc_mod = _ilu.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(_gc_mod)
_gc_mod.ensure_gc_kernel_path()

import torch  # noqa: E402
import habana_frameworks.torch  # noqa: E402,F401

from vllm_gaudi.models.qwen4exp.h23 import h23_enabled as _h23_enabled  # noqa: E402

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.3.self_attn"
HIDDEN = 2560
PRIME_TOKENS = 256
LONG_PRIME_TOKENS = 2560
OUT_DEFAULT = Path(__file__).resolve().parent / "bench_h23_qsa.json"

_DIST_READY = False
_VLLM_CONFIG = None
_KEEP_CTX = None
_HANDLES: dict = {}
_WEIGHT_MAP: dict | None = None
_CONFIG = None

# config name -> (umbrella, {item: value})
CONFIGS: dict[str, dict] = {
    "off": {"VLLM_QWEN4EXP_QSA_H23": "0"},
    "on": {"VLLM_QWEN4EXP_QSA_H23": "1"},
    "attn": {"VLLM_QWEN4EXP_QSA_H23": "0", "VLLM_QWEN4EXP_QSA_H23_ATTN": "1"},
    "spans": {"VLLM_QWEN4EXP_QSA_H23": "0", "VLLM_QWEN4EXP_QSA_H23_SPANS": "1"},
    "tile": {"VLLM_QWEN4EXP_QSA_H23": "0", "VLLM_QWEN4EXP_QSA_H23_TILE": "1"},
    "fuse": {"VLLM_QWEN4EXP_QSA_H23": "0", "VLLM_QWEN4EXP_QSA_H23_FUSE": "1"},
    "fuse_nomerge": {"VLLM_QWEN4EXP_QSA_H23": "0", "VLLM_QWEN4EXP_QSA_H23_FUSE": "1",
                     "VLLM_QWEN4EXP_QSA_H23_MERGE": "0"},
    # H28 items (H23 stays at its default, so these are H28-only deltas).
    "h28_off": {"VLLM_QWEN4EXP_QSA_H23": "1", "VLLM_QWEN4EXP_QSA_H28": "0"},
    "h28_sparse": {"VLLM_QWEN4EXP_QSA_H23": "1", "VLLM_QWEN4EXP_QSA_H28": "0",
                   "VLLM_QWEN4EXP_QSA_H28_SPARSE": "1"},
    "h28_rope": {"VLLM_QWEN4EXP_QSA_H23": "1", "VLLM_QWEN4EXP_QSA_H28": "0",
                 "VLLM_QWEN4EXP_QSA_H28_ROPE": "1"},
}
H23_ENV = (
    "VLLM_QWEN4EXP_QSA_H23",
    "VLLM_QWEN4EXP_QSA_H23_ATTN",
    "VLLM_QWEN4EXP_QSA_H23_SPANS",
    "VLLM_QWEN4EXP_QSA_H23_TILE",
    "VLLM_QWEN4EXP_QSA_H23_FUSE",
    "VLLM_QWEN4EXP_QSA_H23_MERGE",
    "VLLM_QWEN4EXP_QSA_EXACT",
    "VLLM_QWEN4EXP_QSA_H28",
    "VLLM_QWEN4EXP_QSA_H28_SPARSE",
    "VLLM_QWEN4EXP_QSA_H28_ROPE",
)


# --------------------------------------------------------------------------
# checkpoint / model scaffolding (bench_layer_e2e + test_qsa_layer_hpu recipe)
# --------------------------------------------------------------------------
def _weight_map() -> dict[str, str]:
    global _WEIGHT_MAP
    if _WEIGHT_MAP is None:
        _WEIGHT_MAP = json.loads(
            (CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
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


def _ensure_dist():
    global _DIST_READY, _VLLM_CONFIG, _KEEP_CTX, _CONFIG
    if _DIST_READY:
        return _VLLM_CONFIG
    import vllm_gaudi  # noqa: F401  (registers the HPU platform first)

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    cfg = Qwen4ExpTextConfig.from_dict(text)
    _CONFIG = cfg
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
    _KEEP_CTX.__enter__()
    initialize_model_parallel(1, 1)
    _DIST_READY = True
    _VLLM_CONFIG = vllm_config
    return vllm_config


def _cfg():
    _ensure_dist()
    return _CONFIG


def _layer_state() -> dict[str, torch.Tensor]:
    return {
        "qkv_proj.weight": torch.cat((
            _dequant_mxfp8(f"{LAYER}.q_proj"),
            _dequant_mxfp8(f"{LAYER}.k_proj"),
            _dequant_mxfp8(f"{LAYER}.v_proj"),
        ), dim=0).contiguous(),
        "q_norm.weight": _load(f"{LAYER}.q_norm.weight"),
        "k_norm.weight": _load(f"{LAYER}.k_norm.weight"),
        "o_proj.weight": _dequant_mxfp8(f"{LAYER}.o_proj"),
        "indexer.index_qk_proj.weight": _dequant_mxfp8(f"{LAYER}.indexer.index_qk_proj"),
        "indexer.q_layernorm.weight": _load(f"{LAYER}.indexer.q_layernorm.weight"),
        "indexer.k_layernorm.weight": _load(f"{LAYER}.indexer.k_layernorm.weight"),
    }


def build_layer(variant: str, slot: int = 3):
    """``plain`` = Qwen4ExpQSAAttention, ``quant`` = QuantQwen4ExpQSAAttention.

    ``slot`` is the layer index used in the module prefix; it only exists so
    two variants can be built in one process (vLLM's ``Attention`` rejects a
    duplicate layer name). The weights are always Layer 3's.

    ``quant`` is the class ``Qwen4ExpDecoderLayer`` builds in production; with
    the ``qwen4exp_plain`` load there is no quant config, so it keeps the plain
    ``index_qk_proj`` (that is exactly the production shape of this layer after
    Step 1's plain load).
    """
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

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
    torch.manual_seed(0)
    if variant == "quant":
        from vllm_gaudi.models.qwen4_exp import QuantQwen4ExpQSAAttention

        layer = QuantQwen4ExpQSAAttention(
            vllm_config=vllm_config, config=cfg, layer_id=slot,
            quant_config=None, prefix=f"model.layers.{slot}.self_attn",
        )
    else:
        from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention

        layer = Qwen4ExpQSAAttention(
            vllm_config=vllm_config, config=cfg, layer_id=slot,
            prefix=f"model.layers.{slot}.self_attn",
        )
    state = _layer_state()
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not [m for m in missing if "rotary" not in m and "attn." not in m], missing
    assert not unexpected, unexpected
    layer.to(torch.bfloat16)
    return layer.to("hpu"), state


# --------------------------------------------------------------------------
# timing helpers (bench_layer_e2e semantics)
# --------------------------------------------------------------------------
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
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _checksum(out) -> float:
    if isinstance(out, (tuple, list)):
        return sum(_checksum(o) for o in out)
    return float(out.float().sum().item())


def time_sync(fn, warmup: int, samples: int) -> tuple[float, float]:
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


def time_phase(setup, fn, warmup: int, samples: int) -> float:
    """Time ``fn`` alone, with ``setup()`` re-run untimed before every call.

    The QSA phases mutate the software cache (``commit`` appends, the pooling
    comes with it), so a per-phase attribution number needs the cache restored
    before each call. ``setup`` does that *and* replays the phases the target
    consumes; the ``_sync_step()`` after ``setup`` drains it, so the timed
    window holds the target phase and nothing else.
    """
    for _ in range(warmup):
        setup()
        with torch.no_grad():
            fn()
        _sync_step()
    times = []
    for _ in range(samples):
        setup()
        _sync_step()  # drain the setup: not part of the measured phase
        t0 = time.perf_counter()
        with torch.no_grad():
            out = fn()
        _sync_step()
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    return _median(times)


def time_unsync(fn, reps: int, samples: int) -> tuple[float, float]:
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
        torch.hpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3 / reps)
    check = float(acc.float().item())
    assert check == check and abs(check) != float("inf"), f"non-finite checksum {check}"
    return _median(times), check


def time_chain(fn, reset, reps: int, samples: int) -> tuple[float, float]:
    check = 0.0
    for _ in range(2):
        reset()
        with torch.no_grad():
            check += _checksum(fn())
        _sync_step()
    times = []
    for _ in range(samples):
        reset()
        t0 = time.perf_counter()
        for _ in range(reps):
            with torch.no_grad():
                out = fn()
        _sync_step()
        times.append((time.perf_counter() - t0) * 1e3 / reps)
        check += _checksum(out)
    assert check == check and abs(check) != float("inf"), f"non-finite chain checksum {check}"
    return _median(times), check


# --------------------------------------------------------------------------
# per-phase attribution: a faithful re-spelling of Qwen4ExpQSAAttention.forward
# --------------------------------------------------------------------------
def make_phases(layer, positions, hidden_states, query_start_loc):
    """Ordered ``(name, fn)`` phases; the output lands in ``st['out']``.

    Every phase calls the production helper the forward calls, and the caller
    asserts the phased output is bit-identical to ``layer(...)``, so this
    cannot silently drift from the real path.
    """
    from vllm_gaudi.models.qwen4exp.qsa import (
        _slice_positions,
        _step_spans,
        clamp_qsa_indices,
        dense_prefix_attention,
        select_qsa_indices_from_pooled,
        sparse_attention,
    )

    st: dict = {"num_tokens": hidden_states.shape[0]}

    def proj():
        merged = layer._merged_projection_weight(hidden_states)
        st["merged"] = merged
        if merged is not None:
            projected = torch.nn.functional.linear(hidden_states, merged)
            st["qkv"] = projected[:, : layer._qkv_rows]
            st["index_qk"] = projected[:, layer._qkv_rows:]
        else:
            qkv = layer.qkv_proj(hidden_states)
            st["qkv"] = qkv[0] if isinstance(qkv, tuple) else qkv
            st["index_qk"] = None

    def qk_norm_rope():
        st["q"], st["k"], st["v"], st["gate"] = layer._project_qkv_gate(st["qkv"], positions)

    def index_norm_rope():
        if st["index_qk"] is None:
            st["q_idx"], st["raw_k"] = layer.indexer.project_qk(
                hidden_states, positions, layer.rotary_emb)
        else:
            st["q_idx"], st["raw_k"] = layer.indexer.project_qk_from_proj(
                st["index_qk"], positions, layer.rotary_emb)

    def commit():
        from vllm_gaudi.models.qwen4exp.control import cpu_twin

        num_tokens = st["num_tokens"]
        st["q"] = st["q"].view(num_tokens, layer.num_heads, layer.head_dim)
        st["k"] = st["k"].view(num_tokens, layer.num_kv_heads, layer.head_dim)
        st["v"] = st["v"].view(num_tokens, layer.num_kv_heads, layer.head_dim)
        st["attn"] = hidden_states.new_zeros(num_tokens, layer.num_heads, layer.head_dim)
        cpu_pos = cpu_twin(positions, "positions")
        cpu_qsl = cpu_twin(query_start_loc, "query_start_loc")
        span_pos = cpu_pos if cpu_pos is not None else positions
        span_qsl = cpu_qsl if cpu_qsl is not None else query_start_loc
        spans, starts = _step_spans(num_tokens, span_qsl, span_pos)
        st["slots"] = []
        for slot, (start, end) in enumerate(spans):
            chunk_start = starts[slot] if len(starts) > slot else None
            logical = layer._commit(
                slot,
                _slice_positions(positions, start, end),
                st["raw_k"][start:end],
                st["k"][start:end],
                st["v"][start:end],
                start=chunk_start,
            )
            st["slots"].append((slot, start, end, logical))

    def select():
        import vllm_gaudi.models.qwen4exp.qsa as qsa_mod

        st["indices"] = []
        for slot, start, end, logical in st["slots"]:
            state = layer._bringup[slot]
            s_kv = state.key.shape[0]
            if s_kv <= layer.indexer.token_topk:
                st["indices"].append(None)
                continue
            if qsa_mod._tile_replay_enabled():
                indices = layer._windowed_indices(st["q_idx"][start:end], state, logical)
            else:
                indices = select_qsa_indices_from_pooled(
                    st["q_idx"][start:end],
                    state.pooled,
                    logical,
                    total_blocks=state.pooled.shape[0],
                    token_budget=layer.indexer.token_topk,
                    compress_ratio=layer.indexer.compress_ratio,
                    total_tokens=s_kv,
                    host_min_position=s_kv - (end - start),
                )
            st["indices"].append(clamp_qsa_indices(indices, s_kv))

    def attention():
        from vllm_gaudi.models.qwen4exp.qsa import qsa_static_bucket_enabled

        for slot, start, end, logical in st["slots"]:
            state = layer._bringup[slot]
            k_cache = state.key
            v_cache = state.value
            s_kv = k_cache.shape[0]
            if st["indices"][slot] is None:
                if qsa_static_bucket_enabled(k_cache.device.type) and state.token_capacity > s_kv:
                    st["attn"][start:end] = dense_prefix_attention(
                        st["q"][start:end], state.key_backing, state.value_backing,
                        layer.scaling,
                        query_positions=None if end - start == 1 else logical,
                        valid_num_keys=s_kv)
                else:
                    st["attn"][start:end] = dense_prefix_attention(
                        st["q"][start:end], k_cache, v_cache, layer.scaling,
                        query_positions=None if end - start == 1 else logical)
            else:
                st["attn"][start:end] = sparse_attention(
                    st["q"][start:end], k_cache, v_cache, st["indices"][slot], layer.scaling)

    def out_proj():
        flat = st["attn"].reshape(st["num_tokens"], -1)
        if st["gate"] is not None:
            flat = flat * torch.sigmoid(st["gate"])
        st["out"] = layer.o_proj(flat)
        if isinstance(st["out"], tuple):
            st["out"] = st["out"][0]

    return [
        ("proj", proj),
        ("qk_norm_rope", qk_norm_rope),
        ("index_norm_rope", index_norm_rope),
        ("commit", commit),
        ("select", select),
        ("attention", attention),
        ("o_proj", out_proj),
    ], st


def run_phases(phases) -> torch.Tensor:
    for _name, fn in phases:
        fn()


# --------------------------------------------------------------------------
# regime runners
# --------------------------------------------------------------------------
@dataclass
class RegimeData:
    """Input tensors for one regime, built once and reused by every config."""

    name: str
    mode: str
    tokens: int
    prime: int
    hidden: torch.Tensor
    #: priming chunks ``(positions, hidden, query_start_loc)``; a long context
    #: primes in <= PRIME_CHUNK-token chunks, exactly as the runner would.
    prime_chunks: list


PRIME_CHUNK = 2048


def make_regime_data(gen, name: str, mode: str, tokens: int, prime: int) -> RegimeData:
    def randn(*shape, scale=0.1):
        return (torch.randn(*shape, generator=gen) * scale).bfloat16()

    hidden = randn(tokens, HIDDEN, scale=0.1).to("hpu")
    prime_chunks: list = []
    if mode != "prefill" and prime:
        done = 0
        while done < prime:
            chunk = min(PRIME_CHUNK, prime - done)
            positions = torch.cat(
                [torch.arange(done, done + chunk, device="hpu")] * tokens)
            qsl = torch.tensor(
                [i * chunk for i in range(tokens + 1)], dtype=torch.int32, device="hpu")
            prime_chunks.append((positions, randn(tokens * chunk, HIDDEN, scale=0.1).to("hpu"), qsl))
            done += chunk
    return RegimeData(name, mode, tokens, prime, hidden, prime_chunks)


class Regime:
    """One (name, mode, tokens) cell: priming, a steady-state layer call, reset."""

    def __init__(self, layer, data: RegimeData):
        self.layer = layer
        self.data = data
        self.name = data.name
        self.mode = data.mode
        self.tokens = data.tokens
        self.prime = data.prime
        self.is_prefill = data.mode == "prefill"
        self.hidden = data.hidden
        self.prime_chunks = data.prime_chunks
        self.reprime()

    # -- state ------------------------------------------------------------
    def reprime(self) -> None:
        """Drop the cache and replay the priming under the current config."""
        self.layer.reset_bringup_cache()
        if not self.is_prefill:
            for positions, hidden, qsl in self.prime_chunks:
                self.layer(positions, hidden, query_start_loc=qsl)
        _sync_step()

    def reset(self) -> None:
        """Restore the primed steady state (decode) / clear the cache (prefill)."""
        layer = self.layer
        if self.is_prefill:
            layer.reset_bringup_cache()
            return
        for state in layer._bringup:
            if state is None:
                continue
            state.truncate_tokens(self.prime)
            state.truncate_pooled(self.prime // layer.indexer.compress_ratio)

    # -- callables ---------------------------------------------------------
    def positions(self) -> torch.Tensor:
        """The steady-state positions tensor (a *new* object on every call)."""
        if self.is_prefill:
            return torch.arange(self.tokens, device="hpu")
        return torch.full((self.tokens,), self.prime, dtype=torch.int64, device="hpu")

    def qsl(self) -> torch.Tensor:
        if self.is_prefill:
            return torch.tensor([0, self.tokens], dtype=torch.int32, device="hpu")
        return torch.tensor(list(range(self.tokens + 1)), dtype=torch.int32, device="hpu")

    def layer_call(self, positions=None, hidden=None, qsl=None):
        layer = self.layer
        if self.is_prefill:
            layer.reset_bringup_cache()
        positions = self.positions() if positions is None else positions
        hidden = self.hidden if hidden is None else hidden
        qsl = self.qsl() if qsl is None else qsl
        return layer(positions, hidden, query_start_loc=qsl)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def micro_attribution(layer, tokens_list) -> dict:
    """Per-op costs behind the qk_norm_rope / index_norm_rope phases.

    The two norm phases are 60%+ of a decode-time QSA layer, and the phases are
    atomic (a vendor split+norm+RoPE chain), so this isolates the pieces that
    H23-4 can actually replace: vLLM's GemmaRMSNorm (the main q/k norm), the
    indexer's ``gemma_rmsnorm``, the MRoPE application, and the fused HPU
    ``rms_norm`` on the same shapes.
    """
    from vllm_gaudi.models.qwen4exp import qsa_math

    out: dict = {}
    for tokens in tokens_list:
        q = (torch.randn(tokens, layer.num_heads, layer.head_dim) * 0.5).bfloat16().to("hpu")
        k = (torch.randn(tokens, layer.num_kv_heads, layer.head_dim) * 0.5).bfloat16().to("hpu")
        iq = (torch.randn(tokens, layer.indexer.index_n_heads,
                          layer.indexer.index_head_dim) * 0.5).bfloat16().to("hpu")
        pooled = (torch.randn(640, layer.indexer.index_head_dim) * 0.5).bfloat16().to("hpu")
        positions = torch.arange(tokens, device="hpu")
        qf = q.view(-1, layer.num_heads * layer.head_dim)
        kf = k.view(-1, layer.num_kv_heads * layer.head_dim)

        def t(fn):
            ms, _ = time_sync(fn, warmup=2, samples=7)
            return ms

        row = {
            "vllm_q_norm_ms": t(lambda: layer.q_norm(q)),
            "vllm_k_norm_ms": t(lambda: layer.k_norm(k)),
            "indexer_q_norm_ms": t(lambda: qsa_math.gemma_rmsnorm(
                iq, layer.indexer.q_layernorm.weight,
                layer.indexer.q_layernorm.variance_epsilon)),
            "pooled_norm_ms": t(lambda: qsa_math.gemma_rmsnorm(
                pooled, layer.indexer.k_layernorm.weight,
                layer.indexer.k_layernorm.variance_epsilon)),
            "mrope_qk_ms": t(lambda: layer.rotary_emb(positions, qf, kf)),
            "hpu_rms_norm_q_ms": t(lambda: torch.ops.hpu.rms_norm(
                q, layer.q_norm.weight, layer.q_norm.variance_epsilon, None, True)[0]),
            "hpu_rms_norm_indexer_ms": t(lambda: torch.ops.hpu.rms_norm(
                iq, layer.indexer.q_layernorm.weight,
                layer.indexer.q_layernorm.variance_epsilon, None, True)[0]),
            "hpu_rms_norm_pooled_ms": t(lambda: torch.ops.hpu.rms_norm(
                pooled, layer.indexer.k_layernorm.weight,
                layer.indexer.k_layernorm.variance_epsilon, None, True)[0]),
        }
        out[f"T{tokens}"] = row
        print(f"  micro T={tokens}: " + "  ".join(f"{k.replace('_ms','')}={v:.3f}" for k, v in row.items()),
              flush=True)
    return out



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--variants", default="plain,quant")
    ap.add_argument("--regimes", default="decode_T1,decode_T4,long_decode_T1,prefill_T512")
    ap.add_argument("--configs", default="off,on")
    ap.add_argument("--accuracy-base", default="off",
                    help="config every other config's layer output is compared "
                         "against (default 'off'; H28 runs use 'h28_off')")
    ap.add_argument("--passes", type=int, default=2,
                    help="measurement passes; every pass but the last is warm-up "
                         "(the first execution of a shape on HPU pays recipe/JIT cost)")
    ap.add_argument("--decode-samples", type=int, default=7)
    ap.add_argument("--prefill-samples", type=int, default=3)
    ap.add_argument("--decode-warmup", type=int, default=2)
    ap.add_argument("--prefill-warmup", type=int, default=1)
    ap.add_argument("--chain-reps", type=int, default=3)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for name in configs:
        assert name in CONFIGS, f"unknown config {name!r}; known: {sorted(CONFIGS)}"
    assert args.accuracy_base in configs or not args.json.name.endswith("_ab.json"), \
        "an A/B report needs the accuracy base config"

    lazy = _is_lazy()
    print(f"bench_h23_qsa: lazy={lazy} variants={variants} regimes={regimes} "
          f"configs={configs} passes={args.passes}", flush=True)
    print(f"  GC_KERNEL_PATH={os.environ.get('GC_KERNEL_PATH')}", flush=True)
    gen = torch.Generator().manual_seed(2026)

    layers: dict[str, object] = {}
    for slot, variant in enumerate(variants):
        print(f"building layer variant={variant} (prefix layer {3 + slot}) ...", flush=True)
        layer, _state = build_layer(variant, 3 + slot)
        layers[variant] = layer
    _sync_step()

    results: dict = {
        "benchmark": "bench_h23_qsa (H23 Phase 1)",
        "checkpoint": str(CHECKPOINT),
        "layer": LAYER,
        "pt_hpu_lazy_mode": os.environ.get("PT_HPU_LAZY_MODE"),
        "prime_tokens": PRIME_TOKENS,
        "long_prime_tokens": LONG_PRIME_TOKENS,
        "configs": {name: CONFIGS[name] for name in configs},
        "micro_attribution_ms": {},
        "geometries": {},
        "results": {},
    }
    for variant, layer in layers.items():
        results["geometries"][variant] = {
            "class": type(layer).__name__,
            "qkv_proj_rows": int(layer.qkv_proj.weight.shape[0]),
            "index_qk_proj_rows": int(layer.indexer.index_qk_proj.weight.shape[0]),
            "merged_rows": int(layer.qkv_proj.weight.shape[0]
                               + layer.indexer.index_qk_proj.weight.shape[0]),
            "num_heads": int(layer.num_heads),
            "num_kv_heads": int(layer.num_kv_heads),
            "head_dim": int(layer.head_dim),
            "token_topk": int(layer.indexer.token_topk),
            "compress_ratio": int(layer.indexer.compress_ratio),
            "index_heads": int(layer.indexer.index_n_heads),
            "index_head_dim": int(layer.indexer.index_head_dim),
        }

    def apply_config(name: str) -> None:
        for var in H23_ENV:
            os.environ.pop(var, None)
        os.environ.update(CONFIGS[name])
        from vllm_gaudi.models.qwen4exp.control import (
            invalidate_mrope_cos_sin,
            invalidate_span_plan,
        )

        invalidate_span_plan()
        invalidate_mrope_cos_sin()

    # ---- fixtures: identical inputs for every config ---------------------
    fixtures: dict[str, RegimeData] = {}

    def fixture(regime_name: str) -> RegimeData:
        if regime_name not in fixtures:
            tokens = {"decode_T1": 1, "decode_T4": 4,
                      "long_decode_T1": 1, "prefill_T512": 512}[regime_name]
            mode = "prefill" if regime_name.startswith("prefill") else "decode"
            prime = LONG_PRIME_TOKENS if regime_name.startswith("long") else PRIME_TOKENS
            fixtures[regime_name] = make_regime_data(
                gen, regime_name, mode, tokens, 0 if mode == "prefill" else prime)
        return fixtures[regime_name]

    for regime_name in regimes:
        fixture(regime_name)

    # micro attribution is config independent (op-level shapes)
    apply_config(configs[0])
    for variant, layer in layers.items():
        print(f"micro attribution (variant={variant}) ...", flush=True)
        results["micro_attribution_ms"][variant] = micro_attribution(layer, (1, 4, 512))

    outputs: dict = {}   # (config, variant, regime) -> layer output (float, cpu-ish)
    per_pass: dict[str, list[dict]] = {}
    passes = max(1, args.passes)
    for pass_index in range(passes):
        print(f"\n##### pass {pass_index + 1}/{passes} #####", flush=True)
        for config in configs:
            apply_config(config)
            for variant in variants:
                layer = layers[variant]
                for regime_name in regimes:
                    data = fixture(regime_name)
                    tokens, mode, prime = data.tokens, data.mode, data.prime
                    os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"] = (
                        "1" if regime_name.startswith("long") else "0")
                    samples = args.prefill_samples if mode == "prefill" else args.decode_samples
                    key = f"{config}|{variant}|{regime_name}"
                    cell: dict = {
                        "mode": mode, "tokens": tokens, "prime": prime,
                        "tile_replay": os.environ["VLLM_HPU_PREFILL_TILE_REPLAY"],
                        "checksums": {},
                    }
                    print(f"\n===== pass{pass_index + 1} {key} "
                          f"(mode={mode}, T={tokens}, prime={prime}) =====", flush=True)

                    reg = Regime(layer, data)
                    reg.reset()

                    # The model forward hands every layer of a step the *same*
                    # positions / query_start_loc objects; that identity is what
                    # lets the H23-2 span plan hit for layers 2..12. A per-layer
                    # fresh qsl would hide the whole effect.
                    steady_pos = reg.positions()
                    steady_qsl = reg.qsl()

                    # --- accuracy / drift check of the phased spelling -----
                    pos = steady_pos
                    phases, st = make_phases(layer, pos, reg.hidden, reg.qsl())
                    run_phases(phases)
                    _sync_step()
                    phased_out = st["out"].clone()
                    reg.reset()
                    direct_out = reg.layer_call(positions=pos)
                    _sync_step()
                    reg.reset()
                    assert torch.equal(phased_out, direct_out), (
                        f"{key}: phased spelling drifted from "
                        "Qwen4ExpQSAAttention.forward")
                    outputs[(config, variant, regime_name)] = direct_out.float().clone()

                    # --- merged-projection build cost (H23-4) --------------
                    if (_h23_enabled("fuse") and _h23_enabled("merge")
                            and not lazy):
                        layer.invalidate_h23_merge()
                        t0 = time.perf_counter()
                        built = layer._merged_projection_weight(reg.hidden)
                        torch.hpu.synchronize()
                        cell["merged_projection_build_ms"] = (
                            (time.perf_counter() - t0) * 1e3 if built is not None else None)
                        cell["merged_projection_rows"] = (
                            None if built is None else int(built.shape[0]))
                        layer._merged_projection_weight(reg.hidden)  # warm again
                        _sync_step()
                        print(f"  merged projection build: "
                              f"{cell['merged_projection_build_ms']} ms "
                              f"rows={cell['merged_projection_rows']}", flush=True)
                    else:
                        cell["merged_projection_build_ms"] = None
                        cell["merged_projection_rows"] = None

                    # --- components (sync per phase, setup excluded) ------
                    comps: dict[str, float] = {}
                    for index, (name, phase_fn) in enumerate(phases):
                        def setup(_index=index):
                            reg.reset()
                            for _pname, pfn in phases[:_index]:
                                pfn()

                        comps[name] = time_phase(
                            setup, phase_fn, warmup=1, samples=samples)
                    reg.reset()
                    cell["components_ms"] = comps
                    cell["components_sum_ms"] = sum(comps.values())
                    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in comps.items())
                          + f"  | SUM={cell['components_sum_ms']:.3f} ms", flush=True)

                    # --- full layer, a fresh positions tensor per call -----
                    reps = 1 if mode == "prefill" else 4

                    def full_fresh():
                        # New positions AND qsl objects per call (a new step), so
                        # the H23-2 span plan misses and the per-layer host
                        # extraction is charged -- the standalone-layer number
                        # (what bench_layer_e2e's closure pays).
                        try:
                            return reg.layer_call(positions=reg.positions(),
                                                  qsl=reg.qsl())
                        finally:
                            reg.reset()

                    ms_fresh, check = time_unsync(full_fresh, reps=reps, samples=samples)
                    cell["full_layer_ms"] = ms_fresh
                    cell["checksums"]["full_layer"] = check
                    print(f"  full_layer(fresh pos)      {ms_fresh:8.4f} ms", flush=True)

                    # --- full layer, one positions tensor (warm plan) -----
                    def full_cached():
                        try:
                            return reg.layer_call(positions=steady_pos, qsl=steady_qsl)
                        finally:
                            reg.reset()

                    ms_cached, check = time_unsync(full_cached, reps=reps, samples=samples)
                    cell["full_layer_cached_pos_ms"] = ms_cached
                    cell["checksums"]["full_layer_cached_pos"] = check
                    print(f"  full_layer(cached pos)     {ms_cached:8.4f} ms", flush=True)

                    # --- full layer with the host control bound -----------
                    # Production shape: ``Qwen4ExpModel.forward`` binds one
                    # host copy of positions/input_ids/query_start_loc per step
                    # and every layer reads that twin instead of the device.
                    def full_bound():
                        from vllm_gaudi.models.qwen4exp.control import (
                            bind_host_control,
                            reset_host_control,
                        )

                        token = bind_host_control(steady_pos, None, steady_qsl)
                        try:
                            return reg.layer_call(positions=steady_pos, qsl=steady_qsl)
                        finally:
                            reg.reset()
                            reset_host_control(token)

                    ms_bound, check = time_unsync(full_bound, reps=reps, samples=samples)
                    cell["full_layer_bound_ms"] = ms_bound
                    cell["checksums"]["full_layer_bound"] = check
                    print(f"  full_layer(bound ctrl)     {ms_bound:8.4f} ms", flush=True)

                    # --- 12-layer chained step (decode only) --------------
                    if mode == "decode":
                        def chained_bound():
                            from vllm_gaudi.models.qwen4exp.control import (
                                bind_host_control,
                                reset_host_control,
                            )

                            token = bind_host_control(steady_pos, None, steady_qsl)
                            try:
                                return chained()
                            finally:
                                reset_host_control(token)

                        def chained():
                            sink = None
                            for _ in range(12):
                                out = reg.layer_call(positions=steady_pos,
                                                     qsl=steady_qsl)
                                reg.reset()
                                sink = (out.float().sum() if sink is None
                                        else sink + out.float().sum())
                            return sink

                        ms_chain, check = time_chain(
                            chained, reg.reset, reps=args.chain_reps, samples=samples)
                        cell["chained_12l_step_ms"] = ms_chain
                        cell["chained_12l_per_layer_ms"] = ms_chain / 12.0
                        cell["checksums"]["chained_12l"] = check
                        print(f"  chained_12l_step           {ms_chain:8.4f} ms "
                              f"({ms_chain / 12.0:.4f} ms/layer)", flush=True)

                        ms_chain_b, check = time_chain(
                            chained_bound, reg.reset, reps=args.chain_reps, samples=samples)
                        cell["chained_12l_bound_ms"] = ms_chain_b
                        cell["chained_12l_bound_per_layer_ms"] = ms_chain_b / 12.0
                        cell["checksums"]["chained_12l_bound"] = check
                        print(f"  chained_12l_step (bound)   {ms_chain_b:8.4f} ms "
                              f"({ms_chain_b / 12.0:.4f} ms/layer)", flush=True)

                    per_pass.setdefault(key, []).append(cell)

    # ---- aggregate the passes: per metric keep the *fastest* pass ---------
    # Every pass after the first executes JIT-warm shapes, but the host is a
    # single core and each pass re-primes, so a cell's spread is a few percent.
    # ``min`` over passes is the latency-MAD estimator (a cold pass is only
    # ever slower); ``passes`` keeps the full spread in the JSON.
    def _min_field(cells: list[dict], field: str):
        values = [c[field] for c in cells if c.get(field) is not None]
        return min(values) if values else None

    for key, cells in per_pass.items():
        merged = dict(cells[-1])
        for field in ("full_layer_ms", "full_layer_cached_pos_ms", "full_layer_bound_ms",
                      "chained_12l_step_ms", "chained_12l_per_layer_ms",
                      "chained_12l_bound_ms", "chained_12l_bound_per_layer_ms",
                      "components_sum_ms", "merged_projection_build_ms"):
            merged[field] = _min_field(cells, field)
        comp_names = set()
        for c in cells:
            comp_names.update(c.get("components_ms") or {})
        merged["components_ms"] = {
            comp: min(c["components_ms"][comp] for c in cells
                      if comp in (c.get("components_ms") or {}))
            for comp in sorted(comp_names)
        }
        merged["passes"] = cells
        merged["n_passes"] = len(cells)
        results["results"][key] = merged

    # ---- accuracy: every config vs the base config of the same cell -------
    accuracy = {}
    base_name = args.accuracy_base
    if base_name in configs:
        for config in configs:
            if config == base_name:
                continue
            for variant in variants:
                for regime_name in regimes:
                    base = outputs.get((base_name, variant, regime_name))
                    cur = outputs.get((config, variant, regime_name))
                    if base is None or cur is None:
                        continue
                    diff = (base.detach() - cur.detach()).abs()
                    denom = base.detach().abs().max().clamp(min=1e-9)
                    accuracy[f"{config}|{variant}|{regime_name}"] = {
                        "absmax": float(diff.max()),
                        "rel": float(diff.max() / denom),
                        "bit_identical": bool(torch.equal(base, cur)),
                    }
                    print(f"  accuracy {config}|{variant}|{regime_name} vs "
                          f"{base_name}: absmax={float(diff.max()):.3e} "
                          f"rel={float(diff.max() / denom):.3e} "
                          f"bit_identical={bool(torch.equal(base, cur))}", flush=True)
    else:
        print(f"  accuracy: base config {base_name!r} not in {configs}, skipped",
              flush=True)
    results["accuracy_base"] = base_name
    results["accuracy_vs_off"] = accuracy

    args.json.write_text(json.dumps(results, indent=2) + "\n")

    # ---- printed A/B table ----------------------------------------------
    print("\n### A/B (ms)")
    for regime_name in regimes:
        for variant in variants:
            rows = [results["results"].get(f"{c}|{variant}|{regime_name}")
                    for c in configs]
            if not any(rows):
                continue
            print(f"\n{regime_name} / {variant}   configs: {configs}")
            for field, label in (("full_layer_ms", "full_layer"),
                                 ("full_layer_cached_pos_ms", "full_cached_pos"),
                                 ("full_layer_bound_ms", "full_bound"),
                                 ("chained_12l_step_ms", "chained_12l"),
                                 ("chained_12l_bound_ms", "chained_12l_bound"),
                                 ("components_sum_ms", "components_sum")):
                if not any(r and r.get(field) is not None for r in rows):
                    continue
                values = [r.get(field) if r else None for r in rows]
                print("  " + f"{label:18s}" + " ".join(
                    f"{v:10.4f}" if v is not None else f"{'-':>10s}" for v in values))
            comp_names = list(rows[0]["components_ms"]) if rows[0] else []
            for comp in comp_names:
                values = [r["components_ms"].get(comp) if r else None for r in rows]
                print("  " + f"  {comp:16s}" + " ".join(
                    f"{v:10.4f}" if v is not None else f"{'-':>10s}" for v in values))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
