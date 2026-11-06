# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp (Qwen3.8-Flash-Next) on Gaudi2 - Track D model file.

This is the production-facing model: it composes the working primitives of
``vllm_gaudi/models/qwen4exp/`` (config shim, QSA attention, hyper-connections,
PLE layer/runtime, packed-expert store) and adds the three tracks around them:

* **Track B residency** - routed experts are loaded through
  ``HPUModelOptMixedPrecisionConfig.load_experts`` (or ``modelopt_csf`` when a
  bare MIXED_PRECISION config is built from the checkpoint's ``config.json``):
  packed ``uint8`` planes ``[E, N, K/2]`` + one E4M3 byte per 16 inputs +
  one F32 ``scale2`` per expert, moved to the device with a single ``.to``.
  Nothing ever materializes an expert in BF16 (that would be ~225 GiB).
* **Track C compute** - the decode SwiGLU runs through
  ``custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2`` when that operator is
  registered *and* the tensor geometry satisfies the kernel caps (K % 128,
  N % 4, 1 <= S <= 64, 64 <= E <= 4096 with E % 64). Otherwise the store
  falls back to the shipped fused GEMV on HPU and to the exact
  dequant-then-matmul path on the host. The selected backend is recorded on
  the store and logged once per projection, so a numeric path never changes
  silently.
* **PLE host offload** - the ~26.8 GiB n-gram table (128 shards per PLE
  layer) is *never* a parameter, buffer or device tensor. By default
  (``VLLM_PLE_CPU_OFFLOAD=1``) the shards are mapped read-only from the
  safetensors files and decoded on the host; only the per-token BF16 rows
  cross to the compute device. ``VLLM_PLE_CPU_OFFLOAD=0`` restores the
  legacy pinned-host tables. HBM holds the projections only. A
  :class:`PLEHostOffloader` carries the explicit host-byte budget
  (``VLLM_PLE_HOST_BUDGET_GIB``, default 32 GiB) and the streaming counters
  (rows gathered, host bytes read, bytes sent), so decision E is measurable
  instead of implied.

Quantization dispatch is per prefix (``modelopt_csf.dispatch_kind``): NVFP4
expert groups -> packed store, MXFP8 linears -> ``HPUModelOptMxFp8LinearMethod``
(Gaudi2 per-channel/per-token FP8), W4A16_NVFP4 vision matrices -> the W4A16
BF16-dequant method, everything else -> unquantized. The resolved kind for
every sampled prefix is logged once at model construction
(:func:`quant_dispatch_report`).

The linear-attention half resolves to the plugin's HPU GatedDeltaNet via the
module attribute lookup in ``qwen3_5.py`` (FlashInfer GDN decode,
``hpu_gdn_pytorch``, the Qwen3.8 native QK prep and ``causal_conv1d``);
:func:`resolve_gdn_backend` reports that wiring and the model logs it, so the
identity is checkable without a device.

Environment switches (all optional):

==============================  ===============================================
``VLLM_PLE_CPU_OFFLOAD``        ``1`` (default) mmap host tables, ``0`` legacy
                                pinned tables; the table never enters HBM in
                                either mode.
``VLLM_PLE_HOST_BUDGET_GIB``    host-RAM budget for the n-gram tables, in GiB
                                (default ``32``; the checkpoint needs 26.82).
``VLLM_PLE_HOST_BUDGET_BYTES``  the same budget in bytes; wins when both are
                                set. A table that does not fit raises
                                :class:`PLEHostBudgetExceeded` at attach time.
``VLLM_QWEN4EXP_MIXED_QUANT``   ``1`` (default) honor the checkpoint's
                                MIXED_PRECISION block; ``0`` forces the legacy
                                BF16 load path (``--load-format
                                qwen4exp_plain`` forces it too: that loader
                                dequantizes before the model sees the data).
``VLLM_QWEN4EXP_TRACK_C_MOE``   ``1`` (default) use the Track C GEMV when it is
                                available, ``0`` forces the shipped path.
``VLLM_QWEN4EXP_MOE_LIB``       explicit path to the Track C torch-op library
                                (default: ``<repo>/build/native/``).
``VLLM_QWEN4EXP_EXPERT_SOURCE`` ``auto`` (default), ``source`` (always load the
                                expert groups from the checkpoint index through
                                Track B), ``stream`` (consume them from the
                                weight stream instead).
==============================  ===============================================

Why a module and not ``models/qwen4_exp/`` as well: Python's ``FileFinder``
resolves a directory with ``__init__.py`` before a same-named ``.py`` file, so
adding ``models/qwen4_exp/`` would shadow this required file. The model lives
here in one place and reuses ``models/qwen4exp/`` for every leaf module.
"""

from __future__ import annotations

import json
import os
import struct
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _gdn_mod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm_gaudi.extension.logger import logger as _hpu_logger
from vllm_gaudi.models.qwen4exp import model as _reference
from vllm_gaudi.ops import nvfp4_tpc_layout as tpc_layout
from vllm_gaudi.models.qwen4exp.hyperconnection_amd import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm_gaudi.models.qwen4exp.packed_moe import (
    PackedRoutedExperts,
    apply_routed_experts,
    apply_routed_experts_grouped,
    compact_expert_ids,
    gemv_slots,
    grouped_gemv,
    nvfp4_gemv_enabled,
    prefill_grouped_enabled,
    prefill_grouped_h18,
    prefill_grouped_h22,
    route_topk,
    shared_expert_forward_h15,
    use_grouped_slots,
)
from vllm_gaudi.models.qwen4exp.h24 import h24_enabled as _h24_enabled
from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable
from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer
from vllm_gaudi.models.qwen4exp.ple_runtime import PLENGramEmbedding
from vllm_gaudi.models.qwen4exp.qsa import QSAIndexer, Qwen4ExpQSAAttention

# ``init_logger(__name__)`` (the models/ convention) is silent under ``vllm
# serve``: vLLM configures only its own ``vllm`` logger, so anything below
# ``vllm_gaudi.*`` propagates to a handler-less root at WARNING. The decode-path
# notices (``backends`` / ``note_decode_path``) are the contract that says which
# chain actually ran, so they go through the plugin logger the rest of the HPU
# stack uses (``vllm_gaudi.extension.logger`` -> the configured ``vllm`` logger).
logger = _hpu_logger()

__all__ = [
    "CheckpointTensorSource",
    "GdnBackend",
    "HostPLEEmbedding",
    "HostPLETable",
    "NvFp4ExpertStore",
    "PLEHostBudgetExceeded",
    "PLEHostOffloader",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
    "attach_ple_tables",
    "load_ple_host_embedding",
    "mmap_safetensors_uint8",
    "ple_cpu_offload_enabled",
    "ple_host_budget_bytes",
    "quant_dispatch_report",
    "resolve_gdn_backend",
    "resolve_quant_config",
    "rewrite_quant_namespace",
    "track_c_enabled",
    "track_c_gemv_capability",
    "visual_dispatch_samples",
]

# Transformers v5.18 renamed `qwen_sparse_attention` to `indexed_attention`;
# checkpoints still serialize the plain `full_attention` marker.
_QSA_LAYER_TYPES = ("qwen_sparse_attention", "indexed_attention", "full_attention")

# Missing-tensor tolerance copied from the reference port: activation scales,
# bias placeholders and the KV scales the checkpoint ships but inference
# never reads.
_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint nests the text model under this prefix (same convention as
# qwen4exp/ple_runtime.py). Only the checkpoint reader uses it; the vLLM-side
# module tree is `model.*`.
_CKPT_LANGUAGE_PREFIX = "model.language_model"

# Track C (native/qwen4exp-nvfp4-moe) and the shipped fused GEMV.
_TRACK_C_GEMV = "custom_qwen4exp_nvfp4_moe_gemv_bf16_gaudi2"
_TRACK_C_LIB_ENV = "VLLM_QWEN4EXP_MOE_LIB"
_TRACK_C_SWITCH = "VLLM_QWEN4EXP_TRACK_C_MOE"
_LEGACY_GEMV = "custom_qwen4exp_nvfp4_indexed_gemv_bf16_gaudi2"

# Track C decode GEMV caps, from native/qwen4exp-nvfp4-moe/README.md.
_TRACK_C_MAX_SLOTS = 64
_TRACK_C_MIN_EXPERTS = 64
_TRACK_C_MAX_EXPERTS = 4096


# ---------------------------------------------------------------------------
# Checkpoint reader: index + shards, random access by tensor name
# ---------------------------------------------------------------------------
class CheckpointTensorSource:
    """``keys()``/``get_tensor()`` source over ``model.safetensors.index.json``.

    Track B's loaders take any object exposing those two methods, so this is
    the only checkpoint-shape knowledge the model needs. It accepts both
    layouts produced by ``tools/qwen4exp_flashnext/convert.py`` (shards under
    ``tensors/``) and a plain ``save_pretrained`` directory. Fall back to a
    header scan when the index is missing so a half-converted directory still
    loads.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self._weight_map = dict(json.loads(index.read_text())["weight_map"])
        else:
            self._weight_map = self._scan_headers()
        if not self._weight_map:
            raise FileNotFoundError(f"no safetensors tensors under {self.root}")
        self._handles: dict[Path, Any] = {}

    # -- discovery ---------------------------------------------------------
    def _shard_paths(self) -> list[Path]:
        tensors = sorted((self.root / "tensors").glob("*.safetensors"))
        return tensors or sorted(self.root.glob("*.safetensors"))

    def _scan_headers(self) -> dict[str, str]:
        weight_map: dict[str, str] = {}
        for shard in self._shard_paths():
            with open(shard, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            for name in header:
                if name != "__metadata__":
                    weight_map[name] = shard.name
        return weight_map

    def shard_path(self, name: str) -> Path:
        shard = Path(self._weight_map[name]).name
        candidate = self.root / "tensors" / shard
        return candidate if candidate.is_file() else self.root / shard

    # -- Track B source protocol ------------------------------------------
    def keys(self) -> list[str]:
        return list(self._weight_map)

    def __contains__(self, name: str) -> bool:
        return name in self._weight_map

    def get_tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.shard_path(name)
        if shard not in self._handles:
            self._handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(name)


def _checkpoint_root(vllm_config: VllmConfig) -> Path | None:
    """Local checkpoint directory, or None when the model was not a directory."""
    model = getattr(vllm_config.model_config, "model", None)
    if not model:
        return None
    root = Path(str(model))
    return root if root.is_dir() else None


# ---------------------------------------------------------------------------
# Quantization: MIXED_PRECISION resolution for the checkpoint's prefixes
# ---------------------------------------------------------------------------
def _checkpoint_quant_dict(vllm_config: VllmConfig) -> dict[str, Any] | None:
    """The checkpoint's ``quantization_config`` mapping, wherever it survived."""
    hf_config = vllm_config.model_config.hf_config
    for holder in (hf_config, getattr(hf_config, "text_config", None)):
        quant = getattr(holder, "quantization_config", None)
        if isinstance(quant, Mapping) and quant:
            return dict(quant)
    # The plugin's PretrainedConfig facade nulls quantization_config for the
    # legacy BF16 path, but the parsed shim keeps the raw dict.
    shim = getattr(hf_config, "_shim", None)
    quant = getattr(shim, "quantization_config", None) if shim is not None else None
    if isinstance(quant, Mapping) and quant:
        return dict(quant)
    root = _checkpoint_root(vllm_config)
    if root is not None and (root / "config.json").is_file():
        payload = json.loads((root / "config.json").read_text())
        quant = payload.get("quantization_config")
        if isinstance(quant, Mapping) and quant:
            return dict(quant)
    return None


def _mixed_quant_enabled() -> bool:
    raw = os.environ.get("VLLM_QWEN4EXP_MIXED_QUANT", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _loader_dequantizes_at_load(vllm_config: VllmConfig) -> bool:
    """``--load-format qwen4exp_plain`` streams BF16 by construction.

    That loader dequantizes MXFP8 and NVFP4 groups before the model sees
    them, so it cannot feed the F8/U8 parameters a MIXED_PRECISION model
    declares. Mixing the two would copy BF16 values into FP8 slots, so the
    format wins and the model stays on the legacy BF16 path.
    """
    load_config = getattr(vllm_config, "load_config", None)
    return str(getattr(load_config, "load_format", "")) == "qwen4exp_plain"


#: Namespace of the checkpoint's ``quantized_layers`` keys. ``mtp.`` is
#: dropped outright: the draft model is not part of this module tree.
_CKPT_QUANT_NAMESPACES = ("model.language_model.", "model.visual.", "mtp.")


def rewrite_quant_namespace(config: Any, mapper: Any, root_prefix: str = "") -> int:
    """Move ``quantized_layers`` keys into this model's module namespace.

    ``config.json`` names layers after the checkpoint (``model.language_model.
    layers.0...``) while vLLM builds modules under the *mapped* tree
    (``language_model.model.layers.0...``). The per-prefix dispatch compares
    module prefixes, so the declaration has to follow the same mapping as the
    weights -- otherwise every layer looks unquantized and the checkpoint's
    packed planes would be read as BF16.

    Idempotent: once no key carries a checkpoint namespace the config is left
    alone, so the outer model (which rewrites first) wins.
    """
    quantized = getattr(config, "quantized_layers", None)
    if not quantized or mapper is None:
        return 0
    if getattr(config, "_qwen4exp_quant_namespace", None) is not None:
        # Already rewritten for this module tree (an enclosing model ran
        # first, or load_weights is being called twice).
        return 0
    if not any(name.startswith(_CKPT_QUANT_NAMESPACES) for name in quantized):
        return 0
    prefix = root_prefix.strip(".")

    def rename(name: str) -> str | None:
        mapped = mapper.apply_dict({name: None})
        if not mapped:  # mapped to None: dropped by the model's mapper
            return None
        new_name = next(iter(mapped))
        return f"{prefix}.{new_name}" if prefix else new_name

    rewritten: dict[str, Any] = {}
    for name, info in quantized.items():
        new_name = rename(name)
        if new_name is not None:
            rewritten[new_name] = info
    config.quantized_layers = rewritten
    excluded = list(getattr(config, "exclude_modules", None) or [])
    if excluded:
        # Ignore patterns match by substring/wildcard, not by module path, so
        # they keep the checkpoint namespace (mtp.* entries are dropped).
        config.exclude_modules = [
            mapped for mapped in (next(iter(mapper.apply_dict({n: None})), None) for n in excluded)
            if mapped is not None
        ]
    config._qwen4exp_quant_namespace = prefix or "model"
    return len(rewritten)


def resolve_quant_config(
    vllm_config: VllmConfig,
    *,
    mapper: Any | None = None,
    root_prefix: str = "",
) -> Any | None:
    """The quantization config this model must build its layers with.

    ``vllm_config.quant_config`` wins. Otherwise, when the checkpoint declares
    ``quant_algo: MIXED_PRECISION`` and the switches allow it, a ModelOpt mixed
    config is built from the checkpoint dict and installed on ``vllm_config``
    so the HPU GDN layer (which reads it from the config, not from an
    argument) dispatches its MXFP8 projections through Track B as well.
    """
    existing = vllm_config.quant_config
    if existing is not None:
        if mapper is not None:
            rewrite_quant_namespace(existing, mapper, root_prefix)
        return existing
    if not _mixed_quant_enabled():
        return None
    if _loader_dequantizes_at_load(vllm_config):
        logger.info("qwen4_exp: --load-format qwen4exp_plain dequantizes at load; "
                    "staying on the legacy BF16 path (no MIXED_PRECISION dispatch)")
        return None
    quant = _checkpoint_quant_dict(vllm_config)
    if not quant or str(quant.get("quant_algo", "")).upper() != "MIXED_PRECISION":
        return None

    from vllm.model_executor.layers.quantization import get_quantization_config

    # Importing the HPU config registers the "modelopt_mixed" method and the
    # HPU platform allow-list entry; ordering matters.
    import vllm_gaudi.ops.hpu_modelopt  # noqa: F401

    quant_cls = get_quantization_config("modelopt_mixed")
    config = quant_cls.from_config(quant)
    rewritten = rewrite_quant_namespace(config, mapper, root_prefix)
    vllm_config.quant_config = config
    logger.info(
        "qwen4_exp: MIXED_PRECISION quantization enabled from the checkpoint "
        "(%d declared layer prefixes, %d after namespace mapping)",
        len(quant.get("quantized_layers") or {}), rewritten)
    return config


# ---------------------------------------------------------------------------
# Per-prefix quantization report (logged, unit-tested)
# ---------------------------------------------------------------------------
def quant_dispatch_samples(stem: str, layer_types: Sequence[str]) -> dict[str, str]:
    """Representative module prefixes for each quantization class.

    ``stem`` is the model tree prefix (``language_model.model`` under the
    multimodal wrapper). The GDN and QSA samples point at the *first* layer of
    that type rather than layer 0, so a checkpoint whose layer 0 is the other
    kind still samples a real projection.
    """
    samples = {
        "experts": f"{stem}.layers.0.mlp.experts",
        "shared_expert": f"{stem}.layers.0.mlp.shared_expert.gate_proj",
    }
    if "linear_attention" in layer_types:
        index = list(layer_types).index("linear_attention")
        samples["gdn"] = f"{stem}.layers.{index}.linear_attn.in_proj_qkv"
    qsa_layers = [index for index, kind in enumerate(layer_types) if kind in _QSA_LAYER_TYPES]
    if qsa_layers:
        samples["qsa"] = f"{stem}.layers.{qsa_layers[0]}.self_attn.q_proj"
    return samples


def visual_dispatch_samples(quant_config: Any) -> dict[str, str]:
    """The vision tower's MIXED_PRECISION classes: one MXFP8 + one W4A16 prefix.

    Taken from the checkpoint's own declarations because the tower is built by
    vLLM's ``Qwen3_VisionTransformer``, not by this module. The W4A16 class is
    exactly what has to be dequantized to BF16 at load (decision D).
    """
    declared = list(getattr(quant_config, "quantized_layers", None) or ())
    samples: dict[str, str] = {}
    for prefix in declared:
        if ".visual." not in prefix and not prefix.startswith("visual."):
            continue
        if prefix.endswith(".qkv") and "visual_attn_qkv" not in samples:
            samples["visual_attn_qkv"] = prefix
        elif prefix.endswith(".mlp.linear_fc1") and "visual_mlp_fc1" not in samples:
            samples["visual_mlp_fc1"] = prefix
        if {"visual_attn_qkv", "visual_mlp_fc1"} <= samples.keys():
            break
    return samples


def quant_dispatch_report(
    quant_config: Any,
    *,
    stem: str,
    layer_types: Sequence[str],
    extras: Mapping[str, str] | None = None,
    tag: str = "language",
    include_language: bool = True,
) -> dict[str, str]:
    """Resolve the Gaudi2 method kind for every sampled prefix and log it.

    The single logging point for decision D: one line per model construction
    that states which storage/compute class each part of the checkpoint took.
    Returns the ``name -> kind`` mapping so a test can assert on it without
    parsing the log. ``include_language=False`` reports only ``extras`` (the
    vision tower's own prefixes).
    """
    from vllm_gaudi.ops import modelopt_csf

    samples = quant_dispatch_samples(stem, layer_types) if include_language else {}
    if extras:
        samples.update(dict(extras))
    kinds = {
        name: modelopt_csf.dispatch_kind(
            prefix,
            getattr(quant_config, "quantized_layers", {}) or {},
            getattr(quant_config, "exclude_modules", []) or [],
            getattr(quant_config, "packed_modules_mapping", {}) or {},
        )
        for name, prefix in samples.items()
    }
    logger.info("qwen4_exp: MIXED_PRECISION dispatch (%s) %s", tag, kinds)
    return kinds


# ---------------------------------------------------------------------------
# GDN identity: the linear half is the plugin's HPU GatedDeltaNet
# ---------------------------------------------------------------------------
#: Modules the plugin GDN forward is built from. ``symbol=None`` only checks
#: that the module imported; a named symbol must exist on it.
_GDN_OP_MODULES: tuple[tuple[str, str | None], ...] = (
    ("flashinfer_gaudi.gdn_decode", "gated_delta_rule_mtp_packed"),
    ("vllm_gaudi.ops.hpu_gdn_pytorch", "hpu_chunk_gated_delta_rule"),
    ("vllm_gaudi.ops.causal_conv1d_pytorch", "hpu_causal_conv1d_fwd_native"),
    ("vllm_gaudi.models.qwen3_5", "load_qwen38_native_qk_prep"),
)


@dataclass(frozen=True)
class GdnBackend:
    """What ``linear_attention`` layers actually are on this host."""

    module: str
    class_name: str
    is_plugin: bool
    ops: dict[str, str]
    import_error: str | None = None

    @property
    def identity(self) -> str:
        return f"{self.module}.{self.class_name}"

    def describe(self) -> str:
        error = f", import_error={self.import_error}" if self.import_error else ""
        return (f"{self.identity} (plugin={self.is_plugin}, "
                f"ops={self.ops}{error})")


def resolve_gdn_backend() -> GdnBackend:
    """Report the class the decoder's GDN lookup resolves to and its ops.

    The decoder reads ``QwenGatedDeltaNetAttention`` off the vendor module at
    construction time; importing ``vllm_gaudi.models.qwen3_5`` rebinds that
    attribute to ``HPUGatedDeltaNetAttention`` (the model registry does that
    import in production). This helper performs the same import *before*
    reading the attribute, so the reported identity is the one a layer built
    now would get. It never constructs a layer and never touches an HPU
    tensor.
    """
    import importlib

    import_error = None
    plugin_cls = None
    try:
        plugin_cls = getattr(importlib.import_module("vllm_gaudi.models.qwen3_5"),
                             "HPUGatedDeltaNetAttention", None)
    except Exception as exc:  # pragma: no cover - only without the plugin
        import_error = f"{type(exc).__name__}: {exc}"
        logger.warning("qwen4_exp: vllm_gaudi.models.qwen3_5 did not import (%s); "
                       "linear_attention would fall back to the vendor GDN class", import_error)
    # Read the attribute only after the import: it is rebound at import time.
    target = getattr(_gdn_mod, "QwenGatedDeltaNetAttention")
    ops: dict[str, str] = {}
    for module_name, symbol in _GDN_OP_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - host without the op libs
            ops[module_name] = f"unavailable ({type(exc).__name__})"
            continue
        if symbol is None or hasattr(module, symbol):
            ops[module_name] = "loaded"
        else:
            ops[module_name] = f"missing {symbol}"
    return GdnBackend(
        module=target.__module__,
        class_name=target.__name__,
        is_plugin=plugin_cls is not None and target is plugin_cls,
        ops=ops,
        import_error=import_error,
    )


# ---------------------------------------------------------------------------
# PLE host-resident n-gram tables
# ---------------------------------------------------------------------------
def ple_cpu_offload_enabled() -> bool:
    """``VLLM_PLE_CPU_OFFLOAD``: 1 (default) mmap host tables, 0 legacy pinned."""
    raw = os.environ.get("VLLM_PLE_CPU_OFFLOAD")
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    """Parse one shard's JSON header, cached by (path, size, mtime).

    The real checkpoint puts ~15k tensors in one shard, so a 9 MiB header is
    normal. A 128-shard PLE table maps 256 tensors out of 7 shards: parsing
    that header per tensor would read ~2.4 GiB of JSON before the first token.
    """
    path = Path(path)
    stat = path.stat()
    return _safetensors_header_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _safetensors_header_cached(path: str, size: int, mtime_ns: int) -> tuple[int, dict[str, Any]]:
    del size, mtime_ns  # cache key only
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return 8 + length, json.loads(handle.read(length))


#: Explicit host-RAM budget for the PLE n-gram tables (decision E). The
#: checkpoint's 256 shard tensors are 26.82 GiB, so 32 GiB leaves ~5 GiB of
#: headroom for the page-cache working set and per-token decode buffers.
PLE_DEFAULT_HOST_BUDGET_BYTES = 32 << 30
PLE_HOST_BUDGET_GIB_ENV = "VLLM_PLE_HOST_BUDGET_GIB"
PLE_HOST_BUDGET_BYTES_ENV = "VLLM_PLE_HOST_BUDGET_BYTES"


class PLEHostBudgetExceeded(RuntimeError):
    """The host-resident n-gram tables do not fit the configured budget."""


def ple_host_budget_bytes() -> int:
    """The host-byte budget for the PLE tables.

    ``VLLM_PLE_HOST_BUDGET_BYTES`` wins over ``VLLM_PLE_HOST_BUDGET_GIB``;
    neither set means :data:`PLE_DEFAULT_HOST_BUDGET_BYTES` (32 GiB). The
    budget is explicit because the alternative — discovering at run time that
    the host cannot hold ~26.8 GiB of tables — happens after the card has been
    claimed.
    """
    raw_bytes = os.environ.get(PLE_HOST_BUDGET_BYTES_ENV, "").strip()
    if raw_bytes:
        value = int(raw_bytes)
        if value <= 0:
            raise ValueError(f"{PLE_HOST_BUDGET_BYTES_ENV} must be positive, got {value}")
        return value
    raw_gib = os.environ.get(PLE_HOST_BUDGET_GIB_ENV, "").strip()
    if raw_gib:
        value = float(raw_gib)
        if value <= 0:
            raise ValueError(f"{PLE_HOST_BUDGET_GIB_ENV} must be positive, got {value}")
        return int(value * 2**30)
    return PLE_DEFAULT_HOST_BUDGET_BYTES


@dataclass
class PLEHostOffloader:
    """Decision E's accounting: host tables, per-token rows, explicit budget.

    One instance per model. ``reserve`` is called once per PLE layer from the
    shard headers *before* the tables are mapped, so a checkpoint that would
    not fit fails before the card starts serving; ``record_lookup`` is called
    from the single host/device transfer point, so the measured traffic is the
    real one:
    ``rows_gathered`` counts ``tokens x num_heads`` row gathers,
    ``host_bytes_read`` counts only those rows' packed + scale bytes, and
    ``device_bytes_sent`` counts only the decoded BF16 rows.
    """

    budget_bytes: int = field(default_factory=ple_host_budget_bytes)
    mode: str = "mmap"
    reserved: dict[int, int] = field(default_factory=dict)
    lookups: int = 0
    rows_gathered: int = 0
    host_bytes_read: int = 0
    device_bytes_sent: int = 0

    def __post_init__(self) -> None:
        if int(self.budget_bytes) <= 0:
            raise ValueError(f"PLE host budget must be positive, got {self.budget_bytes}")

    @property
    def used_bytes(self) -> int:
        return sum(self.reserved.values())

    @property
    def remaining_bytes(self) -> int:
        return int(self.budget_bytes) - self.used_bytes

    def reserve(self, layer_idx: int, nbytes: int) -> None:
        """Charge one PLE layer's tables against the budget.

        Re-attaching the same layer with the same size is a no-op (``attach``
        may run twice: at load and from an explicit call); a different size
        for an already-charged layer is a bug, not a resize.
        """
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError(f"cannot reserve {nbytes} host bytes")
        layer_idx = int(layer_idx)
        previous = self.reserved.get(layer_idx)
        if previous is not None:
            if previous != nbytes:
                raise RuntimeError(f"PLE layer {layer_idx} already reserved {previous} bytes, "
                                   f"re-reserving {nbytes}")
            return
        if self.used_bytes + nbytes > int(self.budget_bytes):
            raise PLEHostBudgetExceeded(
                f"PLE layer {layer_idx} needs {nbytes / 2**30:.2f} GiB but only "
                f"{self.remaining_bytes / 2**30:.2f} GiB remain of the "
                f"{int(self.budget_bytes) / 2**30:.2f} GiB host budget "
                f"(raise {PLE_HOST_BUDGET_GIB_ENV} if the host really has the memory)")
        self.reserved[layer_idx] = nbytes

    def record_lookup(self, *, rows: int, host_bytes: int, device_bytes: int) -> None:
        self.lookups += 1
        self.rows_gathered += int(rows)
        self.host_bytes_read += int(host_bytes)
        self.device_bytes_sent += int(device_bytes)

    def report(self) -> str:
        return (f"mode={self.mode} host={self.used_bytes / 2**30:.2f} GiB of "
                f"{int(self.budget_bytes) / 2**30:.2f} GiB budget, "
                f"lookups={self.lookups} rows={self.rows_gathered} "
                f"host_read={self.host_bytes_read} B device_sent={self.device_bytes_sent} B")


def _ngram_host_bytes(embedding: Any) -> int:
    """Resident host bytes of one loaded n-gram embedding (tables only)."""
    tables = getattr(embedding, "tables", None) or []
    return sum(int(table.packed.numel()) + int(table.scales.numel()) for table in tables)


def _ple_layer_host_bytes(source: CheckpointTensorSource, prefix: str) -> int:
    """Byte size of one PLE layer's shards, from safetensors headers only.

    Used to charge the whole model's PLE budget before any file is mapped: a
    checkpoint that does not fit must fail without half a table attached. The
    header parse is cached per shard, so the 256 shard tensors cost 7 JSON
    reads, not 256.
    """
    total = 0
    stem = f"{prefix}.ngram_embedding.shard_"
    headers: dict[Path, dict[str, Any]] = {}
    for name in source.keys():
        if not name.startswith(stem):
            continue
        if not (name.endswith(".weight") or name.endswith(".weight_scale")):
            continue
        shard = source.shard_path(name)
        header = headers.get(shard)
        if header is None:
            _, header = _safetensors_header(shard)
            headers[shard] = header
        begin, end = header[name]["data_offsets"]
        total += int(end) - int(begin)
    return total


#: Bytes of PLE table pages locked in RAM so far (``VLLM_PLE_HOST_LOCK``).
_PLE_LOCKED_BYTES = 0


def _lock_host_pages(array: Any) -> None:
    """Fault a mapped table into RAM and keep it there (``VLLM_PLE_HOST_LOCK``, default 1).

    H31: a page-cache-backed table only costs the pages that are looked up, but loading the
    167 GB of other weights pushes about a quarter of the 26.8 GiB n-gram table out of the page
    cache. Every new n-gram in a prompt then waits for a disk read: a fresh 8K prompt took 7.3 s
    to the first token against 4.7 s once the table was resident. ``mlock`` reads the table once
    at load and pins it; the values are unchanged. Without the right to lock (RLIMIT_MEMLOCK) it
    logs and keeps the plain mapping.
    """
    global _PLE_LOCKED_BYTES
    if os.environ.get("VLLM_PLE_HOST_LOCK", "1").strip().lower() in ("0", "false", "off", "no"):
        return
    nbytes = int(array.nbytes)
    if nbytes == 0:
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if libc.mlock(ctypes.c_void_p(array.ctypes.data), ctypes.c_size_t(nbytes)) != 0:
        err = ctypes.get_errno()
        logger.warning("qwen4_exp: mlock of a %.1f MiB PLE table failed (%s); it stays a plain mapping",
                       nbytes / 2**20, os.strerror(err))
        return
    _PLE_LOCKED_BYTES += nbytes


def mmap_safetensors_uint8(path: str | Path, name: str) -> torch.Tensor:
    """Map one ``U8``/``F8_E4M3`` tensor read-only instead of copying it.

    The returned tensor is backed by the page cache: a 28.8 GiB table costs
    only the pages that are actually looked up. Only byte-sized tensors are
    accepted; that covers the packed nibbles and the E4M3 scale planes.
    """
    import numpy as np

    path = Path(path)
    data_start, header = _safetensors_header(path)
    entry = header[name]
    dtype = entry["dtype"]
    if dtype not in ("U8", "F8_E4M3"):
        raise ValueError(f"{name}: {dtype} is not a byte tensor; cannot map read-only")
    begin, end = entry["data_offsets"]
    flat = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start + begin, shape=(end - begin, ))
    _lock_host_pages(flat)
    with warnings.catch_warnings():
        # A read-only mapping is the point; torch only warns about the
        # missing writable flag.
        warnings.simplefilter("ignore", UserWarning)
        tensor = torch.from_numpy(flat).view(*tuple(entry["shape"]))
    return tensor


class HostPLETable:
    """One host-resident PLE shard: mmap-backed packed rows + scale rows."""

    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        *,
        global_scale: float = 1.0,
        backing: Any = None,
    ) -> None:
        if packed.ndim != 2 or scales.ndim != 2:
            raise ValueError("packed and scales must be 2-D [rows, cols]")
        rows, width = packed.shape
        if width * 2 != scales.shape[1] * 16:
            raise ValueError(f"scale columns ({scales.shape[1]}) do not cover {width * 2} values")
        if scales.shape[0] != rows:
            raise ValueError("packed and scales row counts differ")
        self.rows = int(rows)
        self.embed_dim = int(width) * 2
        if packed.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("PLE shard tables must stay in host memory")
        self.packed = packed.contiguous()
        if scales.dtype != torch.uint8:
            scales = scales.view(torch.uint8)
        self.scales = scales.contiguous()
        self.global_scale = float(global_scale)
        # Optional owner handle for the mapping (the tensor itself already
        # keeps the numpy buffer alive). A plain attribute, never a buffer, so
        # the table can be neither moved nor serialized with the module.
        self.backing = backing

    def lookup_packed(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.device.type != "cpu":
            ids = ids.cpu()
        index = ids.long()
        return self.packed.index_select(0, index), self.scales.index_select(0, index)

    def lookup(
        self,
        ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        packed_rows, scale_rows = self.lookup_packed(ids)
        decoded = PLEShardTable.dequant(packed_rows, scale_rows, self.global_scale, dtype, None)
        return _transfer_host_rows(decoded, device)


def _transfer_host_rows(rows: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
    """Move the decoded per-token rows to the compute device.

    The single transfer point for PLE data: the host tests patch this symbol
    to prove that only ``tokens x heads x head_dim`` ever crosses, never a
    table.
    """
    if device is None:
        return rows
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if rows.device == device:
        return rows
    return rows.to(device)


class HostPLEEmbedding(PLENGramEmbedding):
    """``PLENGramEmbedding`` whose shard tables live in host RAM.

    The hash/shuffle/routing logic is reused unchanged; only the transfer is
    routed through :func:`_transfer_host_rows` so the host/device boundary is
    one auditable call site. Every lookup reports to the model's
    :class:`PLEHostOffloader`, so the streaming claim is measured rather than
    asserted: one gather per token (``tokens x num_heads`` rows), and only the
    decoded ``tokens x num_heads x head_dim`` values cross to the device.
    """

    host_mode = "mmap"

    def __init__(self, *args: Any, offloader: PLEHostOffloader | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.offloader = offloader
        self.layer_idx = None if layer_idx is None else int(layer_idx)

    @property
    def row_bytes(self) -> int:
        """Packed + scale bytes for one row of one shard."""
        return self.head_dim // 2 + self.head_dim // 16

    def host_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Every tensor this embedding keeps in host memory."""
        for index, table in enumerate(self.tables):
            yield f"shard_{index}.packed", table.packed
            yield f"shard_{index}.scales", table.scales
        yield "layer_multipliers", self.layer_multipliers
        yield "head_vocab_sizes", self.head_vocab_sizes
        yield "head_offsets", self.head_offsets

    def assert_host_resident(self) -> int:
        """Raise unless every table/metadata tensor is a host CPU tensor."""
        offenders = [f"{name}={tensor.device}" for name, tensor in self.host_tensors()
                     if tensor.device.type != "cpu"]
        if offenders:
            raise RuntimeError("PLE n-gram state left host memory: " + ", ".join(offenders))
        return sum(1 for _ in self.host_tensors())

    def lookup(
        self,
        ids: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decoded ``[T, num_heads * head_dim]`` rows for ``ids``.

        Default path: gather packed rows on the host, decode NVFP4 through the
        cached-LUT path **on the host**, and cross the host/device boundary
        once with the decoded rows (``_transfer_host_rows``).

        H24-1b (``VLLM_QWEN4EXP_PLE_H24_H2D=1``) instead stages the 90 B
        packed rows into the pinned buffer, moves them asynchronously and
        decodes on the destination device — 3.5x fewer bytes over the bus.
        It is **off by default because it lost**: the device decode is 12-14
        eager launches, and on this card that costs more than the extra bytes
        it saves at every measured regime. It also
        changes what the offloader accounts for (packed bytes cross, not
        decoded bytes), so the host-offload tests pin it off.
        """
        target = None if device is None else torch.device(device)
        if (
            target is not None
            and target.type == "hpu"
            and _h24_enabled("h2d")
        ):
            rows = super().lookup(ids, dtype=dtype, device=target)
            if self.offloader is not None:
                num_rows = int(ids.shape[0]) * int(ids.shape[1]) if ids.ndim == 2 else int(ids.numel())
                self.offloader.record_lookup(
                    rows=num_rows,
                    host_bytes=num_rows * self.row_bytes,
                    device_bytes=num_rows * self.row_bytes,
                )
            return rows
        rows = super().lookup(ids, dtype=dtype, device=None)
        moved = _transfer_host_rows(rows, device)
        if self.offloader is not None:
            num_rows = int(ids.shape[0]) * int(ids.shape[1]) if ids.ndim == 2 else int(ids.numel())
            self.offloader.record_lookup(
                rows=num_rows,
                host_bytes=num_rows * self.row_bytes,
                device_bytes=int(moved.numel()) * int(moved.element_size()),
            )
        return moved

    @property
    def host_bytes(self) -> int:
        return _ngram_host_bytes(self)


def _ple_metadata(source: CheckpointTensorSource, prefix: str) -> tuple[torch.Tensor, ...]:
    multipliers = source.get_tensor(f"{prefix}.layer_multipliers")
    offsets = source.get_tensor(f"{prefix}.ngram_heads_offsets")
    sizes = source.get_tensor(f"{prefix}.ngram_heads_vocab_sizes")
    scale_name = f"{prefix}.ngram_embedding.weight_scale_2"
    global_scale = float(source.get_tensor(scale_name).reshape(-1)[0]) if scale_name in source else 1.0
    return multipliers, offsets, sizes, global_scale


def _ple_shard_ids(source: CheckpointTensorSource, prefix: str) -> list[int]:
    stem = f"{prefix}.ngram_embedding.shard_"
    return sorted({
        int(name[len(stem):].split(".", 1)[0])
        for name in source.keys()
        if name.startswith(stem) and name.endswith(".weight") and "scale" not in name
    })


def load_ple_host_embedding(
    checkpoint_dir: str | Path,
    *,
    layer_idx: int,
    heads_per_ngram: int,
    ngram_size: int,
    offload: bool | None = None,
    language_prefix: str = _CKPT_LANGUAGE_PREFIX,
    source: CheckpointTensorSource | None = None,
    offloader: PLEHostOffloader | None = None,
) -> PLENGramEmbedding:
    """Load one PLE layer's n-gram tables into host RAM.

    ``offload`` True (default, ``VLLM_PLE_CPU_OFFLOAD=1``) maps the shard
    payloads read-only; False restores the legacy pinned copies. Both modes
    keep the table off the compute device. When ``offloader`` is given the
    layer's host bytes are charged against its budget (idempotently: attaching
    the same layer twice with the same size is allowed).
    """
    offload = ple_cpu_offload_enabled() if offload is None else offload
    root = Path(checkpoint_dir)

    if not offload:
        # Legacy mode: the shipped loader copies every shard into pinned host
        # memory. Still host-only, just 26.8 GiB of it instead of mmap pages.
        from vllm_gaudi.models.qwen4exp.ple_runtime import load_ple_ngram

        table = load_ple_ngram(
            root,
            layer_idx=layer_idx,
            heads_per_ngram=heads_per_ngram,
            ngram_size=ngram_size,
            language_prefix=language_prefix,
            pin_memory=True,
        )
        table.host_mode = "pinned"  # type: ignore[attr-defined]
        if offloader is not None:
            offloader.mode = "pinned"
            offloader.reserve(layer_idx, _ngram_host_bytes(table))
            table.offloader = offloader  # type: ignore[attr-defined]
        logger.info("qwen4_exp: PLE layer %d tables pinned in host RAM (legacy mode)", layer_idx)
        return table

    source = source if source is not None else CheckpointTensorSource(root)
    prefix = f"{language_prefix}.layers.{layer_idx}.ple.ple_embedding"
    multipliers, offsets, sizes, global_scale = _ple_metadata(source, prefix)
    shard_ids = _ple_shard_ids(source, prefix)
    if not shard_ids:
        raise FileNotFoundError(f"no PLE shards under {prefix}")

    tables = []
    for shard_id in shard_ids:
        base = f"{prefix}.ngram_embedding.shard_{shard_id}"
        weight_path = source.shard_path(f"{base}.weight")
        scale_path = source.shard_path(f"{base}.weight_scale")
        packed = mmap_safetensors_uint8(weight_path, f"{base}.weight")
        scales = mmap_safetensors_uint8(scale_path, f"{base}.weight_scale")
        tables.append(HostPLETable(packed, scales, global_scale=global_scale))
    embedding = HostPLEEmbedding(
        tables,
        multipliers,
        sizes,
        offsets,
        heads_per_ngram=heads_per_ngram,
        ngram_size=ngram_size,
        shard_rows=tables[0].rows,
        offloader=offloader,
        layer_idx=layer_idx,
    )
    embedding.assert_host_resident()
    if offloader is not None:
        offloader.reserve(layer_idx, embedding.host_bytes)
    logger.info(
        "qwen4_exp: PLE layer %d tables mapped read-only in host RAM "
        "(%.1f MiB, 0 bytes on the device)", layer_idx, embedding.host_bytes / 2**20)
    return embedding


def attach_ple_tables(
    language_model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    offloader: PLEHostOffloader | None = None,
) -> int:
    """Attach host-resident n-gram tables to every PLE layer. Returns layers.

    The whole model's host-byte demand is charged *before* the first mapping,
    so a checkpoint that does not fit the budget fails without a half-attached
    table set. The offloader is stored on the language model for
    introspection (``model.ple_offloader``) and reused when the loader attaches
    tables automatically at load time.
    """
    global _PLE_LOCKED_BYTES
    # A second attach maps the same file pages again; they are shared, so count one attach.
    _PLE_LOCKED_BYTES = 0
    root = Path(checkpoint_dir)
    source = CheckpointTensorSource(root)
    if not any(".ple.ple_embedding." in name for name in source.keys()):
        logger.warning("qwen4_exp: checkpoint ships no PLE n-gram tables; skipping attach")
        return 0
    offloader = offloader if offloader is not None else PLEHostOffloader()
    ple_layers = [layer for layer in language_model.model.layers if getattr(layer, "ple", None) is not None]
    for layer in ple_layers:
        prefix = f"{_CKPT_LANGUAGE_PREFIX}.layers.{layer.ple.layer_idx}.ple.ple_embedding"
        offloader.reserve(layer.ple.layer_idx, _ple_layer_host_bytes(source, prefix))
    for layer in ple_layers:
        ple = layer.ple
        ple.attach_ngram(
            load_ple_host_embedding(
                root,
                layer_idx=ple.layer_idx,
                heads_per_ngram=ple.heads_per_ngram,
                # conv dilation == ngram_size in this architecture
                ngram_size=ple.short_conv_dilation,
                source=source,
                offloader=offloader,
            ))
    language_model._ple_offloader = offloader
    logger.info("qwen4_exp: PLE host offload: %d layer(s), %s, locked in RAM=%.2f GiB", len(ple_layers),
                offloader.report(), _PLE_LOCKED_BYTES / 2**30)
    return len(ple_layers)


# ---------------------------------------------------------------------------
# Track C: NVFP4 decode GEMV operator (native/qwen4exp-nvfp4-moe)
# ---------------------------------------------------------------------------
def _custom_op(name: str) -> Any | None:
    namespace = getattr(torch.ops, "custom_op", None)
    return getattr(namespace, name, None) if namespace is not None else None


def _track_c_switch_on() -> bool:
    """``VLLM_QWEN4EXP_TRACK_C_MOE`` (default ``1``)."""
    return os.environ.get(_TRACK_C_SWITCH, "1").strip().lower() not in ("0", "false", "off", "no")


def track_c_enabled() -> bool:
    """Whether Track C owns the decode MoE: switch on *and* operator loaded.

    Nothing in the serve path calls :func:`ensure_track_c_library`, so Track C
    is off in production and the fused indexed GEMV (H8/H10/H15) serves
    decode. Unit tests that register a Track C operator flip this to True and
    keep the per-projection Track C contract.
    """
    return _track_c_switch_on() and _custom_op(_TRACK_C_GEMV) is not None


def _track_c_candidates() -> Iterator[Path]:
    override = os.environ.get(_TRACK_C_LIB_ENV)
    if override:
        yield Path(override)
    here = Path(__file__).resolve()
    for base in (here.parents[3], Path.cwd()):
        yield from base.glob("build/native/hpu_qwen4exp_nvfp4_moe_pt2*.so")


def ensure_track_c_library() -> bool:
    """Load the Track C torch-op library. Returns True when the op is usable.

    Never raises: a missing or unloadable library simply selects the fallback,
    which is logged by the store on first use.
    """
    if _custom_op(_TRACK_C_GEMV) is not None:
        return True
    for candidate in _track_c_candidates():
        if not candidate.is_file():
            continue
        try:
            # The HPU dispatch key only exists once the Habana extension is
            # imported; this is guarded (and only reached off the CPU path).
            import habana_frameworks.torch  # noqa: F401

            torch.ops.load_library(str(candidate))
        except Exception as exc:  # pragma: no cover - host without HPU libs
            logger.warning("qwen4_exp: could not load Track C library %s: %s", candidate, exc)
            continue
        if _custom_op(_TRACK_C_GEMV) is not None:
            logger.info("qwen4_exp: Track C NVFP4 MoE operators loaded from %s", candidate)
            return True
    return False


def track_c_gemv_capability(
    packed: torch.Tensor,
    scales: torch.Tensor,
    x: torch.Tensor,
    ids: torch.Tensor,
    *,
    op_available: bool | None = None,
) -> tuple[bool, str]:
    """Whether the Track C GEMV can serve this call, and why not.

    The caps are the ones the host GlueCode enforces, checked here so a
    geometry mismatch falls back instead of aborting a real serve.
    """
    if op_available is None:
        op_available = _custom_op(_TRACK_C_GEMV) is not None
    if not op_available:
        return False, "operator not registered"
    if packed.numel() == 0:
        return False, "expert planes not loaded"
    if x.dtype != torch.bfloat16:
        return False, f"activations must be BF16, got {x.dtype}"
    if x.ndim != 2 or ids.ndim != 1:
        return False, f"activations {tuple(x.shape)} / ids {tuple(ids.shape)} must be 2-D / 1-D"
    experts, rows, half = packed.shape
    hidden = half * 2
    slots = x.shape[0]
    if x.shape[1] != hidden:
        return False, f"activation width {x.shape[1]} != K {hidden}"
    if slots != ids.numel():
        return False, f"{slots} activation rows for {ids.numel()} expert ids"
    if hidden % 128:
        return False, f"K={hidden} is not a multiple of 128"
    if rows % 4:
        return False, f"N={rows} is not a multiple of 4"
    if not 1 <= slots <= _TRACK_C_MAX_SLOTS:
        return False, f"S={slots} outside 1..{_TRACK_C_MAX_SLOTS}"
    if not _TRACK_C_MIN_EXPERTS <= experts <= _TRACK_C_MAX_EXPERTS or experts % 64:
        return False, f"E={experts} outside the 64-multiple range {_TRACK_C_MIN_EXPERTS}..{_TRACK_C_MAX_EXPERTS}"
    if scales.shape != (experts, rows, half // 8):
        return False, f"scale plane {tuple(scales.shape)} != {(experts, rows, half // 8)}"
    if x.device.type == "cpu":
        # The library registers HPU and Meta kernels only; CPU tensors must
        # take the portable path (this is also the offline test path).
        return False, "no Track C kernel for CPU tensors"
    return True, ""


def _legacy_gemv_available() -> bool:
    if _custom_op(_LEGACY_GEMV) is not None:
        return True
    from vllm_gaudi.models.qwen4exp.packed_moe import _gemv_op

    return _gemv_op() is not None


# ---------------------------------------------------------------------------
# Routed experts: Track B residency, Track C decode
# ---------------------------------------------------------------------------
class NvFp4ExpertStore(PackedRoutedExperts):
    """One layer's routed experts, 4-bit packed, loaded through Track B.

    ``PackedRoutedExperts`` already owns the buffer layout and the streamed
    ``load_tensor`` path used by ``--load-format qwen4exp_plain``; this class
    adds the Track B source loader and the projection compute that prefers
    Track C.
    """

    def __init__(self, num_experts: int, *, group_size: int = 16) -> None:
        super().__init__(num_experts)
        self.group_size = int(group_size)
        self.loaded_from: str | None = None
        self.layout: str | None = None  # resident TPC layout (nvfp4_tpc_layout)
        self.geometry: dict[str, Any] | None = None
        self.backends: dict[str, str] = {}
        self._logged: set[tuple[str, str]] = set()

    # -- loading -----------------------------------------------------------
    def load_from_source(self, quant_config: Any, source: CheckpointTensorSource, prefix: str) -> None:
        """Fill every projection from ``quant_config.load_experts`` (Track B).

        The callee decodes CSF scales on the host, in memory (decision A) and
        returns the planes in the resident TPC layout (decision B), so this
        method only checks and moves them.
        """
        device = self.gate_proj_packed.device  # placeholder buffer, created on the model device
        for kind in self._KINDS:
            weights = quant_config.load_experts(
                source,
                prefix,
                self.num_experts,
                projection=kind,
                group_size=self.group_size,
            )
            self._install(kind, weights, device)
        self.loaded_from = f"{prefix} (Track B)"
        logger.info("qwen4_exp: %s loaded packed (%d experts x 3 projections, %s, layout %s)",
                    prefix, self.num_experts, device, self.layout)

    def _install(self, kind: str, weights: Any, device: torch.device) -> None:
        packed = weights.packed
        scales = weights.scales
        scale2 = weights.scale2
        layout = getattr(weights, "layout", None)
        self.layout = layout
        expected_scales = (packed.shape[0], packed.shape[1],
                           tpc_layout.scale_row_bytes_for_layout(packed.shape[2] * 2, layout))
        if packed.ndim != 3 or packed.shape[0] != self.num_experts:
            raise ValueError(f"{kind}: packed plane {tuple(packed.shape)} is not "
                             f"[{self.num_experts}, N, K/2]")
        if tuple(scales.shape) != expected_scales:
            raise ValueError(f"{kind}: scale plane {tuple(scales.shape)} != {expected_scales} "
                             "(one E4M3 byte per 16 inputs)")
        if scale2.numel() != self.num_experts:
            raise ValueError(f"{kind}: scale2 holds {scale2.numel()} values for "
                             f"{self.num_experts} experts")
        report = weights.geometry(require_kernel_caps=False) if hasattr(weights, "geometry") else None
        if report is not None and not report["ok"]:
            logger.warning_once("qwen4_exp: %s planes are not TPC-loadable (%s); the GEMV "
                                "path will not be selected", kind, "; ".join(report["violations"]))
        self.geometry = report
        # Track A residency contract: U8 nibbles + E4M3 scale bytes + F32 one
        # global scale per expert. Nothing here widens a weight to BF16.
        setattr(self, f"{kind}_packed", packed.contiguous().to(device))
        setattr(self, f"{kind}_scale", scales.contiguous().view(torch.uint8).to(device))
        setattr(self, f"{kind}_scale2", scale2.reshape(-1).contiguous().to(device=device, dtype=torch.float32))

    # -- compute -----------------------------------------------------------
    def backend_for(self, kind: str, ids: torch.Tensor, x: torch.Tensor) -> tuple[str, str]:
        """``(backend, reason)`` for one projection call.

        ``track_c`` -> fused Track C GEMV, ``gemv`` -> the shipped fused GEMV
        (HPU only), ``dequant`` -> materialize the selected rows and matmul.
        """
        if not _track_c_switch_on():
            return "dequant", f"{_TRACK_C_SWITCH}=0"
        packed = getattr(self, f"{kind}_packed")
        supported, reason = track_c_gemv_capability(
            packed, getattr(self, f"{kind}_scale"), x, ids)
        if supported:
            return "track_c", ""
        if x.device.type == "hpu" and _legacy_gemv_available():
            return "gemv", f"Track C unavailable ({reason})"
        return "dequant", reason

    def _note(self, kind: str, backend: str, reason: str) -> None:
        if (kind, backend) in self._logged:
            return
        self._logged.add((kind, backend))
        self.backends[kind] = backend
        if backend == "dequant" and reason:
            logger.info_once("qwen4_exp: %s decode uses the dequant path (%s)", kind, reason)
        else:
            logger.info_once("qwen4_exp: %s decode uses backend %r", kind, backend)

    def note_decode_path(self, path: str) -> None:
        """Record the decode chain once (same contract as :meth:`_note`).

        The fused H10/H15 chain never reaches :meth:`forward_proj`, so without
        this the served decode path would change silently -- exactly what the
        module contract forbids.
        """
        if ("decode", path) in self._logged:
            return
        self._logged.add(("decode", path))
        self.backends["decode"] = path
        logger.info_once("qwen4_exp: decode routed experts use %r", path)

    def forward_proj(self, kind: str, ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``x`` [S, K] against the selected experts of one projection."""
        if ids.numel() == 0:
            return x.new_empty((0, getattr(self, f"{kind}_packed").shape[1]), dtype=torch.bfloat16)
        backend, reason = self.backend_for(kind, ids, x)
        self._note(kind, backend, reason)
        if backend == "track_c":
            from vllm_gaudi.models.qwen4exp.packed_moe import ensure_gate_up_rowperm, gate_up_natural

            op = _custom_op(_TRACK_C_GEMV)
            assert op is not None  # capability checked
            ensure_gate_up_rowperm(self)
            out = op(
                ids.to(device=x.device, dtype=torch.int32).contiguous(),
                getattr(self, f"{kind}_packed"),
                getattr(self, f"{kind}_scale"),
                x.contiguous(),
                getattr(self, f"{kind}_scale2"),
            )
            return gate_up_natural(self, out) if kind in ("gate_proj", "up_proj") else out
        if backend == "gemv":
            return gemv_slots(self, kind, ids, x)
        return self._dequant_matmul(kind, ids, x)

    def _dequant_matmul(self, kind: str, ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # scale2 is folded by dequant(), exactly like the fused kernels do, so
        # the fallback does not apply it a second time.
        weight = self.dequant(kind, ids)
        return torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)


def _fused_decode_gemv(hidden: torch.Tensor) -> bool:
    """Whether ``packed_moe.grouped_gemv`` (H10 + H15) can serve this step.

    That chain needs the fused indexed-GEMV extension -- the same one prefill
    already requires -- and a device tensor. On the host only the exact
    dequant path below exists, and it is also the one that records
    ``store.backends`` for the CPU tests.
    """
    if not nvfp4_gemv_enabled() or hidden.device.type != "hpu":
        return False
    try:
        return _legacy_gemv_available()
    except Exception:  # extension missing/unloadable -> per-projection fallback
        return False


def _grouped_swiglu(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    store: NvFp4ExpertStore,
) -> torch.Tensor:
    """Decode SwiGLU over per-token slots.

    Track C owns decode whenever its operator is registered (the per-projection
    contract the Track C tests pin). With Track C off -- the production
    default, because nothing calls :func:`ensure_track_c_library` -- the
    H10 + H15 chain in ``packed_moe.grouped_gemv`` runs the same math in three
    launches: gate/up read ``hidden`` ``[T, H]`` directly (no per-slot
    ``index_select`` replication), one fused SwiGLU GEMV replaces
    gate + up + ``silu * up``, and ``down_epilogue`` replaces the weighted
    ``index_add_``. Both H15 helpers fall back inside ``grouped_gemv`` when an
    op is absent from the extension, and the loop below stays as the host /
    pre-H10 fallback.
    """
    if not track_c_enabled() and _fused_decode_gemv(hidden):
        store.note_decode_path("h10_h15")
        return grouped_gemv(hidden, topk_ids, topk_weights, store)
    tokens, top_k = topk_ids.shape
    ids = topk_ids.reshape(-1).to(dtype=torch.int32).contiguous()
    token_index = torch.arange(tokens, device=hidden.device).repeat_interleave(top_k)
    shared = hidden.index_select(0, token_index)
    gate = store.forward_proj("gate_proj", ids, shared)
    up = store.forward_proj("up_proj", ids, shared)
    activated = nn.functional.silu(gate) * up
    down = store.forward_proj("down_proj", ids, activated)
    mix = topk_weights.reshape(-1, 1).to(dtype=down.dtype)
    return (down * mix).view(tokens, top_k, -1).sum(dim=1)


class Qwen4ExpSparseMoeBlock(_reference.Qwen4ExpSparseMoeBlock):
    """Sparse MoE whose routed experts are the Track B/C packed store."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        _ = reduce_results  # single-card port: no results reduction
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.tp_size = 1
        self.n_routed_experts = int(config.num_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
        self.top_k = int(config.num_experts_per_tok)
        self.is_sequence_parallel = False

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,  # the checkpoint ignores mlp.gate
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = (
            Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,  # MXFP8 -> Track B FP8 PC/PT
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=False,
                prefix=f"{prefix}.shared_expert",
            ) if self.n_shared_experts else None)
        self.experts = NvFp4ExpertStore(self.n_routed_experts)

    @torch._dynamo.disable  # custom TPC dequant / Track C GEMV are not capturable
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, orig_shape[-1])
        if hidden.shape[0] == 0:
            # Graph-capture and bucket warmup probe zero-token steps.
            return hidden_states.view(orig_shape)
        if os.environ.get("VLLM_QWEN4EXP_MOE_LEGACY", "0") == "1":
            # Escape hatch: the shipped reference forward (torch.unique +
            # searchsorted + per-expert loop). It is the exact code path that
            # served end-to-end in earlier rounds, so it is kept reachable when
            # the bucketing variants trip the HPU graph compiler in-engine.
            return _reference.Qwen4ExpSparseMoeBlock.forward(self, hidden_states)

        topk_ids, topk_weights = route_topk(hidden, self.gate.weight, self.top_k)
        if use_grouped_slots(hidden.shape[0], topk_ids.shape[-1]):
            out = _grouped_swiglu(hidden, topk_ids, topk_weights, self.experts)
        else:
            # Prefill: dequantize only the experts this batch touches, then
            # run the three projections as one batched GEMM each over
            # expert-major buckets (Strata's bucket->grouped-GEMM->join shape;
            # VLLM_QWEN4EXP_MOE_PREFILL_GROUPED=0 restores the per-expert loop).
            # H18 (VLLM_QWEN4EXP_MOE_PREFILL_H18=0 disables) does the same
            # three GEMMs one count-ordered expert chunk at a time, so the
            # bucket cap and the BF16 peak are per chunk instead of per layer,
            # and the presence mask / host reads happen once. It returns None
            # in compile-only warmup and on non-HPU tensors, and the H9(a)
            # path below then runs unchanged.
            # H22 (``VLLM_QWEN4EXP_MOE_PREFILL_H22=0``) runs the same three
            # GEMMs projection-first: one dequant launch per projection group,
            # one bucket for the whole layer, each chunk still on its own
            # tight cap. It is bit-identical to H18; H18 stays as the fallback
            # and ``..._PREFILL_H18=0`` still restores the H9(a) path below.
            out = None
            if prefill_grouped_enabled():
                out = prefill_grouped_h22(
                    hidden, topk_ids, topk_weights, self.experts)
            if out is None and prefill_grouped_enabled():
                out = prefill_grouped_h18(
                    hidden, topk_ids, topk_weights, self.experts)
            if out is None:
                unique, local = compact_expert_ids(topk_ids, self.n_routed_experts)
                gate_w = self.experts.dequant("gate_proj", unique)
                up_w = self.experts.dequant("up_proj", unique)
                down_w = self.experts.dequant("down_proj", unique)
                if prefill_grouped_enabled():
                    out = apply_routed_experts_grouped(
                        hidden, local, topk_weights, gate_w, up_w, down_w)
                else:
                    out = apply_routed_experts(
                        hidden, local, topk_weights, gate_w, up_w, down_w)
        if self.shared_expert is not None:
            # H15: merged gate_up + shared_gate GEMM -> fused SwiGLU/sigmoid
            # epilogue -> down GEMM (3 launches). ``shared_expert_forward_h15``
            # returns None -- and the stock MLP runs unchanged -- when the
            # switch is off, the weights are not BF16 (Track B FP8/MXFP8), the
            # shape is not covered, or the tensors are on the host.
            shared = shared_expert_forward_h15(
                self.shared_expert,
                hidden,
                expert_gate=getattr(self, "shared_expert_gate", None),
            )
            out = out + (shared if shared is not None else self.shared_expert(hidden))
        return out.view(orig_shape)


# ---------------------------------------------------------------------------
# QSA attention whose indexer projection takes the checkpoint's quant method
# ---------------------------------------------------------------------------
class _QuantQSAIndexer(QSAIndexer):
    """``QSAIndexer`` with a quantized ``index_qk_proj`` (MXFP8 in this ckpt)."""

    def __init__(self, config, prefix: str = "", quant_config: Any | None = None) -> None:
        super().__init__(config, prefix=prefix)
        if quant_config is None:
            return
        proj = f"{prefix}.index_qk_proj" if prefix else "index_qk_proj"
        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=proj,
        )


class QuantQwen4ExpQSAAttention(Qwen4ExpQSAAttention):
    """``Qwen4ExpQSAAttention`` that routes the indexer through quant dispatch."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_id: int,
        quant_config: Any | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
        if quant_config is not None:
            self.indexer = _QuantQSAIndexer(
                config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                quant_config=quant_config,
            )


def invalidate_h23_merges(model: nn.Module) -> int:
    """Drop cached H23-4 merged QSA projections under ``model``.

    The merged ``[qkv_proj; indexer.index_qk_proj]`` weight is a copy cached
    lazily at the first forward, keyed on the two source weights' storage and
    version. A checkpoint (re)load writes those weights in place through
    ``.data``, which does not bump ``_version``, so the loader drops the cache
    explicitly. Returns how many QSA layers had one.
    """
    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", [])
    dropped = 0
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        invalidate = getattr(attention, "invalidate_h23_merge", None)
        if invalidate is None:
            continue
        if getattr(attention, "_merged_projection_cache", None) is not None:
            dropped += 1
        invalidate()
    return dropped


# ---------------------------------------------------------------------------
# Decoder layer / model
# ---------------------------------------------------------------------------
def _hyper_connection_config(config) -> HyperConnectionConfig:
    return HyperConnectionConfig(
        hc_count=int(config.hc_count),
        hidden_size=int(config.hidden_size),
        params_dtype=torch.bfloat16,
        hc_lowrank=int(config.hc_lowrank),
        rms_norm_eps=float(config.rms_norm_eps),
        hc_per_branch_norm=True,
    )


class Qwen4ExpDecoderLayer(_reference.Qwen4ExpDecoderLayer):
    """Reference decoder layer with the Track B/C MoE and quant-aware QSA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        config=None,
    ) -> None:
        nn.Module.__init__(self)
        # ``config`` lets a caller build a layer against a variant of the
        # text config (the MTP draft strips ple_layer_ids; nothing else).
        if config is None:
            config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.use_hc_sequence_parallel = False
        self.layer_idx = extract_layer_index(prefix)
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)}
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id_map[self.layer_idx + 1],
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            # The plugin replaces this class with HPUGatedDeltaNetAttention;
            # the lookup happens at construction time so the patch applies.
            # GDN reads its projections' quant method from vllm_config.
            self.linear_attn = _gdn_mod.QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            )
        elif layer_type in _QSA_LAYER_TYPES:
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if use_qsa:
                self.self_attn = QuantQwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen4ExpSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

        hc_config = _hyper_connection_config(config)
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )


class Qwen4ExpModel(_reference.Qwen4ExpModel):
    """Reference trunk with the Track D decoder layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.use_hc_sequence_parallel = False
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:  # noqa: F811 - shadows the model prefix
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size)

        from vllm.distributed import get_pp_group

        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = GatedResidual(
                _hyper_connection_config(config),
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None
        self._ple_token_history: list[torch.Tensor] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The reference trunk loader already applies the q/k/v, gate/up and
        # HC stacked mapping, filters the non-persistent PLE buffers and
        # tolerates the checkpoint's scale-only tensors.
        return super().load_weights(weights)


class Qwen4ExpForCausalLM(_reference.Qwen4ExpForCausalLM):
    """Text model: Track B/C experts, host-offloaded PLE, MIXED_PRECISION."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = resolve_quant_config(
            vllm_config, mapper=self.hf_to_vllm_mapper, root_prefix=prefix)
        if self.quant_config is not None:
            # The quant config resolves fused prefixes (qkv/gate_up/kv_proj)
            # through this mapping; without it every merged linear would look
            # unquantized. Union, not replace: an enclosing multimodal model
            # may already have contributed its own fused names.
            mapping = dict(getattr(self.quant_config, "packed_modules_mapping", None) or {})
            mapping.update(self.packed_modules_mapping)
            self.quant_config.packed_modules_mapping = mapping
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        # The H14/H25/H26 GDN paths in the shared qwen3_5 layer were measured on
        # this model: opt in before the layers are built. Other models (the 27B)
        # keep their previous GDN path unless VLLM_QWEN4EXP_GDN_* asks for it.
        from vllm_gaudi.models.qwen4exp.gdn_h14 import enable_for_qwen4exp
        enable_for_qwen4exp()
        # The GDN prefill graph trips a Habana bridge bug on 1.24.1 (see habana_compat).
        from vllm_gaudi.models.qwen4exp.habana_compat import patch_fx_stack_trace
        patch_fx_stack_trace()
        # The linear half must be the plugin's HPU GatedDeltaNet (see the
        # module docstring). Resolve it -- and therefore import qwen3_5, which
        # rebinds the vendor attribute -- *before* the decoder layers look the
        # class up, then verify what was actually built.
        self._gdn_backend = (resolve_gdn_backend()
                             if "linear_attention" in config.layer_types else None)
        if self._gdn_backend is not None:
            logger.info("qwen4_exp: GDN linear attention -> %s", self._gdn_backend.describe())
        self._model_prefix = maybe_prefix(prefix, "model")
        self.model = Qwen4ExpModel(vllm_config=vllm_config,
                                   prefix=self._model_prefix)
        self._gdn_layers_built = self._verify_gdn_identity()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.set_moe_parameters(self.model.layers)

        self._track_b_config = self.quant_config if hasattr(self.quant_config, "load_experts") else None
        self._source = self._open_checkpoint_source(vllm_config)
        self._expert_source_mode = os.environ.get("VLLM_QWEN4EXP_EXPERT_SOURCE", "auto").strip().lower()
        self._experts_from_source = (
            self._track_b_config is not None
            and self._source is not None
            and self._expert_source_mode != "stream"
        )
        self._streamed_expert_tensors = 0
        self.report_quant_dispatch()

    # -- setup helpers -----------------------------------------------------
    def _verify_gdn_identity(self) -> int:
        """Check the built GDN layers are the class :func:`resolve_gdn_backend` named.

        A silent mismatch (the vendor class built because the plugin import
        happened too late) would run a different attention path than the one
        the log claims, so it is a loud warning. Returns the GDN layer count.
        """
        built = [layer for layer in self.model.layers
                 if getattr(layer, "linear_attn", None) is not None]
        if not built:
            return 0
        if self._gdn_backend is not None:
            actual = type(built[0].linear_attn)
            identity = f"{actual.__module__}.{actual.__name__}"
            if identity != self._gdn_backend.identity:
                logger.warning("qwen4_exp: GDN layers are %s but %s was resolved before "
                               "construction; the plugin GDN import may be missing",
                               identity, self._gdn_backend.identity)
        return len(built)

    @staticmethod
    def _open_checkpoint_source(vllm_config: VllmConfig) -> CheckpointTensorSource | None:
        root = _checkpoint_root(vllm_config)
        if root is None:
            return None
        try:
            return CheckpointTensorSource(root)
        except FileNotFoundError:
            return None

    def report_quant_dispatch(self) -> dict[str, str]:
        """Log the per-prefix method the checkpoint resolved to."""
        quant = self.quant_config
        if quant is None:
            return {}
        return quant_dispatch_report(
            quant,
            stem=self._model_prefix,
            layer_types=self.config.layer_types,
            tag="language",
        )

    # -- weight loading ----------------------------------------------------
    def _filter_checkpoint_tensors(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Drop tensors another owner loads (PLE tables, MTP, Track B experts).

        The text-only class has no vision tower, so the checkpoint's
        ``model.visual.*`` tensors are dropped here; before this filter they
        reached ``AutoWeightsLoader`` and aborted with
        "There is no module or parameter named 'visual'".
        """
        for name, tensor in weights:
            if ".ple.ple_embedding." in name:
                continue
            if name.startswith("mtp.") or ".mtp." in name:
                continue
            if name.startswith("visual.") or ".visual." in name:
                continue
            if self._experts_from_source and ".mlp.experts." in name:
                self._streamed_expert_tensors += 1
                continue
            yield name, tensor

    def _load_experts_from_source(self) -> set[str]:
        if not self._experts_from_source:
            return set()
        assert self._source is not None and self._track_b_config is not None
        loaded: set[str] = set()
        for layer in self.model.layers:
            store = getattr(layer.mlp, "experts", None)
            if not isinstance(store, NvFp4ExpertStore):
                continue
            prefix = f"{_CKPT_LANGUAGE_PREFIX}.layers.{layer.layer_idx}.mlp"
            store.load_from_source(self._track_b_config, self._source, prefix)
            loaded.add(f"model.layers.{layer.layer_idx}.mlp.experts")
        return loaded

    def _maybe_attach_ple_tables(self) -> int:
        if not any(layer.ple is not None for layer in self.model.layers):
            return 0
        if self._source is None:
            logger.warning("qwen4_exp: PLE layers found but the checkpoint directory is not "
                           "readable; the n-gram tables cannot be attached")
            return 0
        if os.environ.get("VLLM_QWEN4EXP_PLE_ATTACH", "1").strip().lower() in ("0", "false", "off"):
            return 0
        if getattr(self, "_ple_attached", False):
            return 0
        attached = attach_ple_tables(self, self._source.root, offloader=self.ple_offloader)
        self._ple_attached = attached > 0
        return attached

    def attach_ple_tables(self, checkpoint_dir: str) -> int:
        """Load host-resident n-gram tables onto every PLE layer."""
        attached = attach_ple_tables(self, checkpoint_dir, offloader=self.ple_offloader)
        self._ple_attached = attached > 0
        return attached

    @property
    def ple_offloader(self) -> PLEHostOffloader | None:
        """Decision E accounting for this model's host-resident PLE tables."""
        return getattr(self, "_ple_offloader", None)

    @property
    def ple_host_bytes(self) -> int:
        offloader = self.ple_offloader
        return 0 if offloader is None else offloader.used_bytes

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = self._filter_checkpoint_tensors(weights)
        mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_substr={"mtp.": None})
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(filtered, mapper=mapper)
        loaded |= self._load_experts_from_source()
        self._maybe_attach_ple_tables()
        if self._experts_from_source and self._streamed_expert_tensors:
            logger.info("qwen4_exp: %d streamed expert tensors skipped (Track B source load)",
                        self._streamed_expert_tensors)
        # H23-4: the merged qkv+indexer projection is a *copy* of two weights,
        # cached across forwards. A (re)load writes those weights in place, so
        # the cached concatenation is stale from here on -- drop it.
        invalidated = invalidate_h23_merges(self.model)
        if invalidated:
            logger.info("qwen4_exp: dropped %d cached H23 merged QSA projections", invalidated)
        return loaded


class Qwen4ExpForConditionalGeneration(_reference.Qwen4ExpForConditionalGeneration):
    """Vision + Qwen4Exp language model, both through the Track B dispatch."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = resolve_quant_config(
            vllm_config, mapper=self.hf_to_vllm_mapper, root_prefix=prefix)
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Qwen4ExpForConditionalGeneration requires multimodal_config")
        self.quant_config = quant_config
        if quant_config is not None:
            mapping = dict(getattr(quant_config, "packed_modules_mapping", None) or {})
            mapping.update(self.packed_modules_mapping)
            quant_config.packed_modules_mapping = mapping

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = self._build_visual_tower(config, vllm_config, quant_config, prefix)

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and getattr(self, "visual", None) is not None
            and not isinstance(self.visual, torch.nn.modules.container.ModuleDict))
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                ) for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.set_moe_parameters(self.language_model.model.layers)
        self.report_vision_dispatch()

    def report_vision_dispatch(self) -> dict[str, str]:
        """Log what the vision tower's declared layers resolve to.

        The tower is vLLM's ``Qwen3_VisionTransformer``; its per-prefix class is
        decided by the same MIXED_PRECISION config, so this is the one place
        that states the ``qkv`` (MXFP8), ``proj`` (MXFP8) and W4A16 MLP
        (dequantized to BF16 at load) split for the checkpoint. Empty when the
        tower is not built (``--language-model-only``) or the checkpoint does
        not declare vision layers.
        """
        quant = self.quant_config
        if quant is None or self.language_model_only:
            return {}
        extras = visual_dispatch_samples(quant)
        if not extras:
            return {}
        return quant_dispatch_report(
            quant,
            stem=self.language_model._model_prefix,
            layer_types=self.config.text_config.layer_types,
            extras=extras,
            tag="vision",
            include_language=False,
        )

    def _build_visual_tower(self, config, vllm_config: VllmConfig, quant_config, prefix: str) -> nn.Module:
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

        return Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=config.text_config.rms_norm_eps,
            quant_config=quant_config,  # MXFP8 attn + W4A16 MLP per prefix
            prefix=maybe_prefix(prefix, "visual"),
        )

    def attach_ple_tables(self, checkpoint_dir: str) -> int:
        return self.language_model.attach_ple_tables(checkpoint_dir)

    @property
    def ple_offloader(self) -> PLEHostOffloader | None:
        """Decision E accounting, owned by the language model."""
        return self.language_model.ple_offloader

    @property
    def ple_host_bytes(self) -> int:
        return self.language_model.ple_host_bytes

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_vision = {
            "visual.": None,
            "model.visual.": None,
        } if self.language_model_only else {}
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix=skip_vision,
        )
        loader = AutoWeightsLoader(self, ignore_unexpected_suffixes=_IGNORED_MISSING_SUFFIXES.copy())
        # The language model owns the Track B expert load, the PLE attach and
        # the mtp/PLE-name filtering; AutoWeightsLoader delegates to it.
        return loader.load_weights(weights, mapper=mapper)

