# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Architecture registration + real-config dispatch for the Qwen4Exp port.

Two host-only checks, both well under a second once the plugin is imported:

1. ``Qwen4ExpForConditionalGeneration`` (the checkpoint's declared
   architecture) resolves through vLLM's model registry to
   ``vllm_gaudi.models.qwen4_exp`` -- the Track D class, not the reference
   port -- and that class is a subclass of the reference implementation, so
   the existing tooling keeps working.
2. The real ``config.json`` (162 GiB plain export) parses into the local shim
   and its 545 ``quantized_layers`` declarations dispatch three ways:
   NVFP4 experts/PLE -> packed residency, MXFP8 linears -> Gaudi2 FP8 PC/PT,
   W4A16_NVFP4 vision matrices -> BF16 dequant at load. The ``ignore`` list
   keeps ``lm_head`` in BF16.

Run::

    .venv/bin/python -m pytest -q validation/qwen4exp/test_qwen4_exp_registry.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("VLLM_HW", "cpu")

REPO = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
CONFIG_JSON = CHECKPOINT / "config.json"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# architecture resolution
# ---------------------------------------------------------------------------
def test_architecture_resolves_to_the_track_d_class():
    from vllm.model_executor.models.registry import ModelRegistry

    import vllm_gaudi
    import vllm_gaudi.models as hpu_models

    vllm_gaudi.register_configs()
    hpu_models.register_model()

    assert CONFIG_JSON.is_file(), f"checkpoint config missing: {CONFIG_JSON}"
    architectures = json.loads(CONFIG_JSON.read_text())["architectures"]
    assert architectures == ["Qwen4ExpForConditionalGeneration"], architectures
    arch = architectures[0]

    registered = ModelRegistry.models[arch]
    assert registered.module_name == "vllm_gaudi.models.qwen4_exp"
    assert registered.class_name == "Qwen4ExpForConditionalGeneration"

    model_cls = ModelRegistry._try_load_model_cls(arch)
    from vllm_gaudi.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    assert model_cls is Qwen4ExpForConditionalGeneration
    assert model_cls.__module__ == "vllm_gaudi.models.qwen4_exp"

    # The reference port stays importable and remains the superclass: the
    # plugin's existing Qwen4ExpMTP draft and tools/ tests depend on it.
    from vllm_gaudi.models.qwen4exp.model import (
        Qwen4ExpForConditionalGeneration as ReferenceConditional,
    )

    assert issubclass(model_cls, ReferenceConditional)
    # Text-only architecture also resolves to the Track D language model.
    lm_cls = ModelRegistry._try_load_model_cls("Qwen4ExpForCausalLM")
    assert lm_cls.__module__ == "vllm_gaudi.models.qwen4_exp"


# ---------------------------------------------------------------------------
# real config: parse + 3-way dispatch
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_config() -> dict:
    if not CONFIG_JSON.is_file():
        pytest.skip(f"checkpoint config not present: {CONFIG_JSON}")
    return json.loads(CONFIG_JSON.read_text())


@pytest.fixture(scope="module")
def real_quant_config(real_config):
    import vllm_gaudi  # noqa: F401  (platform + ops registration)

    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm_gaudi.ops import hpu_modelopt  # noqa: F401  (modelopt_mixed)

    config = get_quantization_config("modelopt_mixed").from_config(real_config["quantization_config"])
    assert type(config).__name__ == "HPUModelOptMixedPrecisionConfig"
    return config


def test_real_config_parses_and_structure_matches(real_config):
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpConfig

    root = Qwen4ExpConfig.from_dict(real_config)
    text = root.text
    assert text.num_hidden_layers == 48
    assert text.layer_types.count("full_attention") == 12
    assert text.layer_types.count("linear_attention") == 36
    assert (text.linear_num_key_heads, text.linear_num_value_heads) == (16, 48)
    assert (text.linear_key_head_dim, text.linear_value_head_dim, text.linear_conv_kernel_dim) == (128, 128, 4)
    assert text.num_experts == 512 and text.num_experts_per_tok == 10
    assert text.hc_count == 4
    assert text.ple_layer_ids == [2]
    assert text.indexer_budget == 2048 and text.indexer_compress_ratio == 4

    quant = real_config["quantization_config"]
    assert quant["quant_method"] == "modelopt"
    assert quant["quant_algo"] == "MIXED_PRECISION"
    assert len(quant["quantized_layers"]) == 545
    algorithms = {entry["quant_algo"] for entry in quant["quantized_layers"].values()}
    assert algorithms == {"NVFP4", "MXFP8", "W4A16_NVFP4"}


def test_three_way_dispatch_on_every_declared_prefix(real_quant_config, real_config):
    """Every one of the checkpoint's prefixes maps to a concrete storage kind."""
    from vllm_gaudi.models.qwen4_exp import Qwen4ExpForConditionalGeneration
    from vllm_gaudi.ops import modelopt_csf

    config = real_quant_config
    # The exact mapper the model applies to the weights: config.json names
    # become vLLM module prefixes (including visual.*).
    model_cls_mapper = Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    # config.json names -> vLLM module prefixes (the same mapping the weights
    # use). mtp.* belongs to the draft model and is dropped.
    from vllm_gaudi.models.qwen4_exp import rewrite_quant_namespace

    written = rewrite_quant_namespace(config, model_cls_mapper, root_prefix="")
    assert written == 545 - 2, written  # the two mtp expert groups are dropped
    assert all(name.startswith(("language_model.model.", "visual."))
               for name in config.quantized_layers)

    kinds: dict[str, list[str]] = {}
    for prefix in config.quantized_layers:
        kind = modelopt_csf.dispatch_kind(prefix, config.quantized_layers,
                                          config.exclude_modules, config.packed_modules_mapping)
        kinds.setdefault(kind, []).append(prefix)

    assert {kind: len(prefixes) for kind, prefixes in kinds.items()} == {
        modelopt_csf.KIND_NVFP4_PACKED: 49,   # 48 expert groups + the PLE table
        modelopt_csf.KIND_MXFP8: 438,        # shared/attn/GDN/indexer/vision attn
        modelopt_csf.KIND_W4A16_BF16: 56,     # 27x2 vision MLP + 2 merger
    }

    # Expert groups keep 4-bit residency; a linear must never receive them.
    expert_prefixes = kinds[modelopt_csf.KIND_NVFP4_PACKED]
    assert all(prefix.endswith((".experts", ".ngram_embedding")) for prefix in expert_prefixes)
    assert sum(prefix.endswith(".mlp.experts") for prefix in expert_prefixes) == 48
    assert [p for p in expert_prefixes if p.endswith(".ngram_embedding")] == [
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    ]

    # MXFP8 is exactly the projections the Qwen4Exp modules build linears for.
    tails: dict[str, int] = {}
    for prefix in kinds[modelopt_csf.KIND_MXFP8]:
        tail = prefix.rsplit(".", 1)[-1]
        tails[tail] = tails.get(tail, 0) + 1
    assert tails == {
        # 48 layers x shared expert
        "gate_proj": 48, "up_proj": 48, "down_proj": 48,
        # 36 GDN layers
        "in_proj_a": 36, "in_proj_b": 36, "in_proj_qkv": 36, "in_proj_z": 36, "out_proj": 36,
        # 12 full-attention layers
        "q_proj": 12, "k_proj": 12, "v_proj": 12, "o_proj": 12, "index_qk_proj": 12,
        # vision tower attention (27 blocks)
        "qkv": 27, "proj": 27,
    }, tails

    # W4A16 outside an expert group is a small vision matrix -> BF16 at load.
    w4a16 = kinds[modelopt_csf.KIND_W4A16_BF16]
    assert all(prefix.startswith("visual.") for prefix in w4a16), w4a16[:3]
    assert sum(prefix.endswith("mlp.linear_fc1") for prefix in w4a16) == 27
    assert sum(prefix.endswith("mlp.linear_fc2") for prefix in w4a16) == 27

    # Ignore list: the head and the draft's own pieces stay unquantized.
    assert config.exclude_modules == ["lm_head"]
    assert modelopt_csf.dispatch_kind("language_model.lm_head", config.quantized_layers,
                                      config.exclude_modules,
                                      config.packed_modules_mapping) == modelopt_csf.KIND_UNQUANTIZED
    assert modelopt_csf.dispatch_kind("language_model.model.layers.0.attn_hyper_connection."
                                      "input_mix_weight_up", config.quantized_layers,
                                      config.exclude_modules,
                                      config.packed_modules_mapping) == modelopt_csf.KIND_UNQUANTIZED


# ---------------------------------------------------------------------------
# Track 4: decision E's host-byte budget, measured on the real checkpoint
# ---------------------------------------------------------------------------
def test_real_ple_host_budget_matches_the_checkpoint(real_config):
    """The n-gram tables' host footprint, from the real 162 GiB export.

    Headers only: no shard payload is read, nothing is mapped. The budget the
    offloader enforces (32 GiB by default) must cover the checkpoint's real
    26.82 GiB of PLE tables, and a model whose budget is one byte smaller must
    fail *before* attaching half a table set.
    """
    from vllm_gaudi.models.qwen4_exp import (
        PLEHostBudgetExceeded,
        PLEHostOffloader,
        _ple_layer_host_bytes,
        _safetensors_header,
        CheckpointTensorSource,
        ple_host_budget_bytes,
    )

    if not CHECKPOINT.is_dir():
        pytest.skip(f"checkpoint not present: {CHECKPOINT}")
    text = real_config["text_config"]
    layer_idx = text["ple_layer_ids"][0] - 1  # ple_layer_ids are 1-based
    prefix = f"model.language_model.layers.{layer_idx}.ple.ple_embedding"
    source = CheckpointTensorSource(CHECKPOINT)
    shard_prefix = f"{prefix}.ngram_embedding.shard_"
    names = [name for name in source.keys()
             if name.startswith(shard_prefix)
             and (name.endswith(".weight") or name.endswith(".weight_scale"))]
    assert len(names) == 128 * 2, len(names)
    assert sum(name.endswith(".weight") for name in names) == 128

    total = 0
    row_shapes = set()
    for name in names:
        _, header = _safetensors_header(source.shard_path(name))
        begin, end = header[name]["data_offsets"]
        total += int(end) - int(begin)
        if name.endswith(".weight"):
            row_shapes.add(tuple(header[name]["shape"]))
    rows, packed_cols = next(iter(row_shapes)) if len(row_shapes) == 1 else (None, None)
    assert len(row_shapes) == 1 and packed_cols == 80
    # 128 shards x rows x (80 packed bytes + 10 E4M3 scale bytes).
    assert total == 128 * rows * 90 == _ple_layer_host_bytes(source, prefix)

    num_heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    head_dim = text["ple_embed_dim"] // num_heads
    assert (num_heads, head_dim) == (16, 160)  # tokens x 16 x head_dim per step
    assert packed_cols * 2 == head_dim
    gib = total / 2**30
    assert gib == pytest.approx(26.82, abs=0.01), gib

    budget = ple_host_budget_bytes()
    assert budget == 32 << 30
    offloader = PLEHostOffloader()
    offloader.reserve(layer_idx, total)
    assert offloader.used_bytes == total
    assert offloader.remaining_bytes == budget - total
    with pytest.raises(PLEHostBudgetExceeded, match="host budget"):
        PLEHostOffloader(budget_bytes=total - 1).reserve(layer_idx, total)


# ---------------------------------------------------------------------------
# Track 4: the per-prefix quantization choice is logged for the real config
# ---------------------------------------------------------------------------
def test_real_dispatch_is_logged_for_language_and_vision(real_config, caplog):
    """Every class of the real 545-prefix checkpoint reports its Gaudi2 method.

    Built on a fresh config from the checkpoint's own ``config.json`` so the
    test does not depend on another test's namespace rewrite.
    """
    import vllm_gaudi.ops.hpu_modelopt  # noqa: F401  (registers modelopt_mixed)
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm_gaudi.models.qwen4_exp import (
        Qwen4ExpForConditionalGeneration,
        quant_dispatch_report,
        rewrite_quant_namespace,
        visual_dispatch_samples,
    )
    from vllm_gaudi.ops import modelopt_csf

    config = get_quantization_config("modelopt_mixed").from_config(real_config["quantization_config"])
    config.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    assert rewrite_quant_namespace(
        config, Qwen4ExpForConditionalGeneration.hf_to_vllm_mapper, root_prefix="") == 545 - 2

    with caplog.at_level("INFO", logger="vllm_gaudi.models.qwen4_exp"):
        kinds = quant_dispatch_report(
            config,
            stem="language_model.model",
            layer_types=real_config["text_config"]["layer_types"],
            tag="language",
        )
    assert kinds == {
        "experts": modelopt_csf.KIND_NVFP4_PACKED,
        "shared_expert": modelopt_csf.KIND_MXFP8,
        "gdn": modelopt_csf.KIND_MXFP8,
        "qsa": modelopt_csf.KIND_MXFP8,
    }

    extras = visual_dispatch_samples(config)
    assert set(extras) == {"visual_attn_qkv", "visual_mlp_fc1"}
    assert extras["visual_attn_qkv"].endswith(".attn.qkv")
    assert extras["visual_mlp_fc1"].endswith(".mlp.linear_fc1")
    with caplog.at_level("INFO", logger="vllm_gaudi.models.qwen4_exp"):
        vision = quant_dispatch_report(
            config,
            stem="unused",
            layer_types=(),
            extras=extras,
            tag="vision",
            include_language=False,
        )
    assert vision == {
        "visual_attn_qkv": modelopt_csf.KIND_MXFP8,
        "visual_mlp_fc1": modelopt_csf.KIND_W4A16_BF16,
    }

    messages = [record.getMessage() for record in caplog.records]
    language_lines = [m for m in messages if "MIXED_PRECISION dispatch (language)" in m]
    vision_lines = [m for m in messages if "MIXED_PRECISION dispatch (vision)" in m]
    assert language_lines and vision_lines, messages
    assert modelopt_csf.KIND_NVFP4_PACKED in language_lines[0]
    assert modelopt_csf.KIND_W4A16_BF16 in vision_lines[0]
