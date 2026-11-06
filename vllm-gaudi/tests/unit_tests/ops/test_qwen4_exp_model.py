# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Track D model tests: Qwen4Exp on Gaudi2 without a device.

Everything here runs on CPU tensors with the HPU-only operators replaced:

* the GDN linear-attention layer has no CPU kernel, so its *forward* is
  stubbed for the end-to-end tests. The identity of the layer is NOT stubbed:
  ``test_gdn_linear_attention_is_the_plugin_class_without_a_mock`` builds the
  real ``HPUGatedDeltaNetAttention`` through the production decoder and checks
  its op wiring, and the on-card smoke
  (``validation/qwen4exp/hpu_device_smoke.py``) runs its forward;
* the Track C NVFP4 MoE GEMV is not registered on a host without a device, so
  its *call path* is exercised by patching the operator with the exact
  E2M1 x E4M3 reference (the kernel itself is verified by
  validation/qwen4exp/test_nvfp4_moe_reference.py);
* Track B expert loading, the PLE host tables and the MIXED_PRECISION
  dispatch are exercised for real, on a miniature checkpoint written by the
  fixtures.

What each test proves is stated in its docstring. Run::

    .venv/bin/python -m pytest -q vllm-gaudi/tests/unit_tests/ops/test_qwen4_exp_model.py
"""

from __future__ import annotations

import inspect
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("VLLM_HW", "cpu")
# Per-parameter HPU sync after each weight load needs a driver; this host has
# none. Production keeps the default (true).
os.environ.setdefault("VLLM_WEIGHT_LOAD_FORCE_SYNC", "false")

import vllm_gaudi  # noqa: E402  (plugin load)
import vllm_gaudi.models  # noqa: E402
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm_gaudi.ops import modelopt_csf  # noqa: E402
from vllm_gaudi.ops.hpu_modelopt import (  # noqa: E402
    HPUModelOptMixedPrecisionConfig,
    HPUModelOptMxFp8LinearMethod,
    HPUModelOptW4A16NvFp4LinearMethod,
)

import vllm_gaudi.models.qwen4_exp as qwen4_exp  # noqa: E402
from vllm_gaudi.models.qwen4_exp import (  # noqa: E402
    CheckpointTensorSource,
    HostPLEEmbedding,
    NvFp4ExpertStore,
    PLEHostBudgetExceeded,
    PLEHostOffloader,
    Qwen4ExpForConditionalGeneration,
    load_ple_host_embedding,
    ple_cpu_offload_enabled,
    ple_host_budget_bytes,
    quant_dispatch_report,
    resolve_gdn_backend,
    rewrite_quant_namespace,
    track_c_gemv_capability,
    visual_dispatch_samples,
)
from vllm_gaudi.models.qwen4exp.packed_moe import (  # noqa: E402
    shared_expert_forward_h15,
    shared_expert_h15_parts,
)
from vllm_gaudi.models.qwen4exp import qsa_math  # noqa: E402
from vllm_gaudi.models.qwen4exp.hyperconnection_amd import (  # noqa: E402
    GatedResidual,
    hc_combine,
    hc_gate_mix,
    hc_grouped_gemma_rmsnorm,
    hc_silu,
)
from vllm_gaudi.models.qwen4exp.hyperconnection_base import (  # noqa: E402
    HyperConnectionConfig,
)

vllm_gaudi.register_configs()
vllm_gaudi.models.register_model()

# ---------------------------------------------------------------------------
# Miniature geometry: 4 layers, hidden 256, 4 experts, hc 4, PLE on layer 1,
# a small QSA indexer. Big enough to exercise every shape rule, small enough
# that a forward is milliseconds.
# ---------------------------------------------------------------------------
HIDDEN = 256
MOE_INTER = 64
EXPERTS = 4
TOP_K = 2
HC = 4
PLE_HEADS = 16
PLE_HEAD_DIM = 16
PLE_ROWS_PER_HEAD = 4
PLE_TABLE_ROWS = PLE_HEADS * PLE_ROWS_PER_HEAD
VOCAB = 512

TEXT_CONFIG = {
    "hidden_size": HIDDEN,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "partial_rotary_factor": 0.5,
    "vocab_size": VOCAB,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 128,
    "tie_word_embeddings": False,
    "full_attention_interval": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "indexer_budget": 8,
    "indexer_compress_ratio": 2,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 32,
    "num_experts": EXPERTS,
    "num_experts_per_tok": TOP_K,
    "moe_intermediate_size": MOE_INTER,
    "shared_expert_intermediate_size": MOE_INTER,
    "norm_topk_prob": True,
    "hc_count": HC,
    "hc_lowrank": 16,
    "ple_layer_ids": [2],
    "ple_embed_dim": PLE_HEADS * PLE_HEAD_DIM,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "mtp_num_hidden_layers": 1,
    "eos_token_id": 7,
    "hidden_act": "silu",
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"},
}

VISION_CONFIG = {
    "depth": 1,
    "hidden_size": 32,
    "num_heads": 2,
    "intermediate_size": 64,
    "patch_size": 16,
    "out_hidden_size": HIDDEN,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [],
}

CKPT_LM = "model.language_model"


def _quantized_layers() -> dict[str, dict]:
    """One declaration per storage kind the real checkpoint uses."""
    layer = f"{CKPT_LM}.layers.0"
    qsa_layer = f"{CKPT_LM}.layers.3"
    layers = {
        f"{layer}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
        f"{layer}.mlp.shared_expert.gate_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{layer}.mlp.shared_expert.up_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{layer}.mlp.shared_expert.down_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{qsa_layer}.self_attn.q_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{qsa_layer}.self_attn.k_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{qsa_layer}.self_attn.v_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{qsa_layer}.self_attn.o_proj": {"quant_algo": "MXFP8", "group_size": 32},
        f"{qsa_layer}.self_attn.indexer.index_qk_proj": {"quant_algo": "MXFP8", "group_size": 32},
        "model.visual.blocks.0.attn.qkv": {"quant_algo": "MXFP8", "group_size": 32},
        "model.visual.blocks.0.mlp.linear_fc1": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    }
    # The 36 GDN layers of the real checkpoint declare their five projections
    # MXFP8; the tiny text model has one GDN layer, so declare it here too.
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        layers[f"{layer}.linear_attn.{name}"] = {"quant_algo": "MXFP8", "group_size": 32}
    return layers


def _quant_payload() -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": ["lm_head"],
        "quantized_layers": _quantized_layers(),
    }


# ---------------------------------------------------------------------------
# Fixtures: miniature checkpoint + distributed/model construction
# ---------------------------------------------------------------------------
def write_ple_shard(root: Path) -> None:
    """Write one PLE layer's host tables (metadata + one shard) + the index."""
    from safetensors.torch import save_file

    prefix = f"{CKPT_LM}.layers.1.ple.ple_embedding"
    generator = torch.Generator().manual_seed(11)
    packed = torch.randint(0, 256, (PLE_TABLE_ROWS, PLE_HEAD_DIM // 2), generator=generator, dtype=torch.uint8)
    # E4M3 code 60/61/62 are ordinary 1.5/1.625/1.75 values; 0x7F would be NaN.
    scale_bytes = torch.randint(60, 63, (PLE_TABLE_ROWS, PLE_HEAD_DIM // 16), generator=generator, dtype=torch.uint8)
    tensors = {
        f"{prefix}.layer_multipliers": torch.tensor([3, 5, 7], dtype=torch.int64),
        f"{prefix}.ngram_heads_offsets": torch.arange(0, PLE_TABLE_ROWS, PLE_ROWS_PER_HEAD, dtype=torch.int64),
        f"{prefix}.ngram_heads_vocab_sizes": torch.full((PLE_HEADS, ), PLE_ROWS_PER_HEAD, dtype=torch.int64),
        f"{prefix}.ngram_embedding.weight_scale_2": torch.tensor([0.5], dtype=torch.float32),
        f"{prefix}.ngram_embedding.shard_0.weight": packed,
        f"{prefix}.ngram_embedding.shard_0.weight_scale": scale_bytes.view(torch.float8_e4m3fn),
    }
    shard = root / "tensors" / "model-00001-of-00001.safetensors"
    shard.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": {name: shard.name for name in tensors}}))


def write_tiny_model(root: Path, *, quant: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "text_config": TEXT_CONFIG,
        "vision_config": VISION_CONFIG,
    }
    if quant:
        payload["quantization_config"] = _quant_payload()
    (root / "config.json").write_text(json.dumps(payload))
    write_ple_shard(root)


_DIST_READY = False
_BASE_CONFIG_CTX = None


def _init_dist() -> None:
    global _DIST_READY, _BASE_CONFIG_CTX
    if _DIST_READY:
        return
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    # The vendor custom ops (RMSNorm, ...) instantiate eagerly inside
    # initialize_model_parallel and ask for the current config; keep a bare
    # one installed for the whole session. Per-model tests nest their own.
    _BASE_CONFIG_CTX = set_current_vllm_config(VllmConfig())
    _BASE_CONFIG_CTX.__enter__()
    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    import vllm.platforms as vllm_platforms

    # No driver: the HPU layers that pin state at construction must land on CPU.
    vllm_platforms.current_platform.current_device = lambda: torch.device("cpu")  # type: ignore[method-assign]
    _DIST_READY = True


class FakeGDN(torch.nn.Module):
    """CPU stand-in for ``HPUGatedDeltaNetAttention`` (no CPU kernel).

    Records the hidden width so a test can prove the real decoder geometry was
    built (H=256, HC*H=1024) without running the HPU kernels.
    """

    calls: list[int] = []

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        type(self).calls.append(int(hidden_states.shape[-1]))
        return torch.zeros_like(hidden_states)


@pytest.fixture(scope="session", autouse=True)
def _distributed():
    """gloo world_size=1 so the vendor linear/embedding layers can be built."""
    _init_dist()


@pytest.fixture
def fake_gdn(monkeypatch):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_module

    FakeGDN.calls = []
    monkeypatch.setattr(gdn_module, "QwenGatedDeltaNetAttention", FakeGDN)
    return FakeGDN


def _build_model(root: Path, prefix: str = "model"):
    """Construct the conditional-generation model on CPU for ``root``."""
    # The default dtype is process-global: restore it on the way out so later
    # suites in the same pytest session are not built in bfloat16.
    saved_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model_config = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        dtype="bfloat16",
        seed=0,
        max_model_len=128,
        language_model_only=True,
    )
    vllm_config = VllmConfig(model_config=model_config)
    _init_dist()
    with set_current_vllm_config(vllm_config):
        try:
            model = Qwen4ExpForConditionalGeneration(vllm_config=vllm_config, prefix=prefix)
        finally:
            torch.set_default_dtype(saved_default_dtype)
    # Deterministic weights: the forward assertions compare value-dependent
    # quantities (finiteness, the Track C vs fallback cosine), so an unseeded
    # fill would make the suite flaky rather than the model wrong.
    torch.manual_seed(1234)
    with torch.no_grad():
        for parameter in model.parameters():
            # E4M3 parameters hold real checkpoint bytes in production; a
            # Gaussian fill is only defined for the wider float types.
            if parameter.dtype.is_floating_point and parameter.dtype.itemsize > 1:
                parameter.normal_(0, 0.02)
    return model, vllm_config


def _random_planes(experts: int, rows: int, hidden: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 256, (experts, rows, hidden // 2), generator=generator, dtype=torch.uint8),
        torch.randint(60, 63, (experts, rows, hidden // 16), generator=generator, dtype=torch.uint8),
        torch.rand(experts, generator=generator, dtype=torch.float32) + 0.5,
    )


def fill_expert_planes(model, *, experts: int = EXPERTS, device=None) -> None:
    """Install small NVFP4 planes so the decode path can really run.

    ``device`` lets the on-card smoke (validation/qwen4exp/hpu_device_smoke.py)
    reuse the exact same planes on the device.
    """

    def place(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if device is None else tensor.to(device)

    for layer in model.language_model.model.layers:
        store = layer.mlp.experts
        assert store.gate_proj_packed.numel() == 0
        packed, scales, scale2 = _random_planes(experts, MOE_INTER, HIDDEN, seed=1)
        store.gate_proj_packed, store.gate_proj_scale, store.gate_proj_scale2 = (
            place(packed), place(scales), place(scale2))
        up = _random_planes(experts, MOE_INTER, HIDDEN, seed=2)
        store.up_proj_packed, store.up_proj_scale, _ = (place(up[0]), place(up[1]), place(up[2]))
        store.up_proj_scale2 = store.gate_proj_scale2.clone()  # gate/up share scale2
        down = _random_planes(experts, HIDDEN, MOE_INTER, seed=3)
        store.down_proj_packed, store.down_proj_scale, store.down_proj_scale2 = (
            place(down[0]), place(down[1]), place(down[2]))


# ---------------------------------------------------------------------------
# (1) registry
# ---------------------------------------------------------------------------
def test_architecture_resolves_to_this_module_class():
    """The plugin registers Qwen4ExpForConditionalGeneration to this class."""
    from vllm.model_executor.models.registry import ModelRegistry

    model_cls = ModelRegistry._try_load_model_cls("Qwen4ExpForConditionalGeneration")
    assert model_cls is Qwen4ExpForConditionalGeneration
    assert model_cls.__module__ == "vllm_gaudi.models.qwen4_exp"

    lm_cls = ModelRegistry._try_load_model_cls("Qwen4ExpForCausalLM")
    assert lm_cls is qwen4_exp.Qwen4ExpForCausalLM
    # The reference port stays importable and is the superclass.
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpForConditionalGeneration as Reference

    assert issubclass(model_cls, Reference)


# ---------------------------------------------------------------------------
# (2) MIXED_PRECISION dispatch per prefix
# ---------------------------------------------------------------------------
def _mixed_config() -> HPUModelOptMixedPrecisionConfig:
    from vllm.model_executor.layers.quantization import get_quantization_config

    config = get_quantization_config("modelopt_mixed").from_config(_quant_payload())
    assert isinstance(config, HPUModelOptMixedPrecisionConfig)
    return config


def test_quant_dispatch_selects_the_right_method_per_prefix():
    """Every declared storage kind maps to its Gaudi2 method, nothing else does."""
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    config = _mixed_config()
    prefix = f"{CKPT_LM}.layers.0.mlp.shared_expert"
    merged = MergedColumnParallelLinear(HIDDEN, [MOE_INTER, MOE_INTER], bias=False,
                                        quant_config=config, prefix=f"{prefix}.gate_up_proj")
    assert isinstance(merged.quant_method, HPUModelOptMxFp8LinearMethod)
    # F8 values + one UE8M0 byte per 32 inputs, not a BF16 weight.
    assert merged.weight.dtype == torch.float8_e4m3fn
    assert merged.weight_scale.dtype == torch.uint8

    # gate/up resolve through the model's packed_modules_mapping (fused lookup).
    from vllm_gaudi.ops import modelopt_csf

    merged_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert modelopt_csf.resolve_quant_algo(f"{prefix}.gate_up_proj", config.quantized_layers,
                                           merged_mapping) == "MXFP8"

    vision = ReplicatedLinear(64, 64, bias=False, quant_config=config,
                              prefix="model.visual.blocks.0.mlp.linear_fc1")
    assert isinstance(vision.quant_method, HPUModelOptW4A16NvFp4LinearMethod)
    assert vision.weight.dtype == torch.uint8
    assert vision.weight_scale.dtype == torch.float8_e4m3fn
    assert vision.weight_scale_2.dtype == torch.float32

    unquantized = ReplicatedLinear(HIDDEN, HIDDEN, bias=False, quant_config=config,
                                   prefix=f"{CKPT_LM}.layers.0.attn_hyper_connection.input_mix_weight_up")
    assert isinstance(unquantized.quant_method, UnquantizedLinearMethod)

    # The checkpoint's ignore list ("lm_head") wins over everything else.
    lm_head = ReplicatedLinear(HIDDEN, 16, bias=False, quant_config=config, prefix="lm_head")
    assert isinstance(lm_head.quant_method, UnquantizedLinearMethod)

    # A dense linear may never consume the packed expert planes.
    dense = ReplicatedLinear(HIDDEN, MOE_INTER, bias=False, quant_config=config, prefix="probe")
    with pytest.raises(ValueError, match="load_experts"):
        config.get_quant_method(dense, f"{CKPT_LM}.layers.0.mlp.experts")

    assert modelopt_csf.dispatch_kind(f"{CKPT_LM}.layers.0.mlp.experts", config.quantized_layers,
                                      config.exclude_modules,
                                      merged_mapping) == modelopt_csf.KIND_NVFP4_PACKED


def test_quant_namespace_follows_the_model_tree():
    """config.json names become module prefixes once, and only once."""
    from vllm.model_executor.models.utils import WeightsMapper

    config = _mixed_config()
    mapper = WeightsMapper(orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
        "mtp.": None,
    })
    written = rewrite_quant_namespace(config, mapper, root_prefix="model")
    assert written > 0
    assert f"model.language_model.model.layers.0.mlp.experts" in config.quantized_layers
    assert not any(name.startswith("model.language_model.layers.") for name in config.quantized_layers)
    assert config.exclude_modules == ["lm_head"]
    # Second application is a no-op (the enclosing model rewrote first).
    assert rewrite_quant_namespace(config, mapper, root_prefix="model") == 0
    assert f"model.language_model.model.layers.0.mlp.experts" in config.quantized_layers


def test_gdn_linear_attention_is_the_plugin_class():
    """The linear half is qwen3_5's HPU GDN path, reused unchanged.

    The decoder looks the class up on the vendor module at construction time,
    and the plugin replaces it with ``HPUGatedDeltaNetAttention`` (FlashInfer
    GDN decode, hpu_gdn_pytorch, qwen38 native QK, causal-conv1d). The
    Qwen4Exp GDN dims (H=16, HV=48, K=128, V=128, conv=4) are that path's
    contract, so no specialization is allowed here.
    """
    import vllm_gaudi.models.qwen3_5  # noqa: F401  (installs the monkeypatch)
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_module

    assert qwen4_exp._gdn_mod is gdn_module
    assert gdn_module.QwenGatedDeltaNetAttention is vllm_gaudi.models.qwen3_5.HPUGatedDeltaNetAttention
    from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts

    assert issubclass(NvFp4ExpertStore, PackedRoutedExperts)


def test_plain_loader_disables_mixed_quant():
    """``--load-format qwen4exp_plain`` streams BF16, so it must not meet FP8.

    ``resolve_quant_config`` builds a MIXED_PRECISION config from the
    checkpoint unless the loader dequantizes at load time; otherwise the FP8
    parameters of an MXFP8 layer would receive BF16 values.
    """
    from types import SimpleNamespace

    from vllm.config import LoadConfig

    shim = SimpleNamespace(quantization_config=_quant_payload())
    hf_config = SimpleNamespace(quantization_config=None, _shim=shim)
    model_config = SimpleNamespace(hf_config=hf_config, text_config=None, model="/nonexistent")
    vllm_config = SimpleNamespace(quant_config=None, model_config=model_config,
                                  load_config=LoadConfig(load_format="qwen4exp_plain"))
    assert qwen4_exp._loader_dequantizes_at_load(vllm_config)
    assert qwen4_exp.resolve_quant_config(vllm_config) is None
    assert vllm_config.quant_config is None

    vllm_config.load_config = LoadConfig()  # "auto"
    config = qwen4_exp.resolve_quant_config(vllm_config)
    assert isinstance(config, HPUModelOptMixedPrecisionConfig)
    assert vllm_config.quant_config is config

    # The explicit switch wins as well.
    vllm_config.quant_config = None
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_QWEN4EXP_MIXED_QUANT", "0")
        assert qwen4_exp.resolve_quant_config(vllm_config) is None


# ---------------------------------------------------------------------------
# (3) Track B: packed expert residency
# ---------------------------------------------------------------------------
def _write_expert_group(root: Path, *, experts: int, rows: int, hidden: int) -> dict[str, torch.Tensor]:
    """A ModelOpt-named NVFP4 expert group: weight + plain E4M3 scale + scale2."""
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(71)
    tensors: dict[str, torch.Tensor] = {}
    for kind in ("gate_proj", "up_proj", "down_proj"):
        for expert in range(experts):
            base = f"{CKPT_LM}.layers.0.mlp.experts.{expert}.{kind}"
            tensors[f"{base}.weight"] = torch.randint(0, 256, (rows, hidden // 2),
                                                      generator=generator, dtype=torch.uint8)
            tensors[f"{base}.weight_scale"] = torch.randint(60, 63, (rows, hidden // 16),
                                                            generator=generator,
                                                            dtype=torch.uint8).view(torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_2"] = torch.tensor([0.25 + 0.5 * expert], dtype=torch.float32)
    shard = root / "tensors" / "model-00001-of-00001.safetensors"
    shard.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard.name for name in tensors}}))
    return tensors


def test_track_b_load_keeps_experts_packed(tmp_path):
    """load_experts() fills the resident store byte-for-byte, never in BF16."""
    experts, rows, hidden = 64, 8, 128
    root = tmp_path / "expert-ckpt"
    on_disk = _write_expert_group(root, experts=experts, rows=rows, hidden=hidden)
    source = CheckpointTensorSource(root)
    store = NvFp4ExpertStore(experts)

    config = _mixed_config()
    store.load_from_source(config, source, f"{CKPT_LM}.layers.0.mlp")

    assert store.loaded_from.endswith("(Track B)")
    # The resident layout the Track 3 GEMV declares is recorded on the store,
    # and the planes the loader installed satisfy its row geometry.
    from vllm_gaudi.ops import nvfp4_tpc_layout as tpc_layout

    assert store.layout == tpc_layout.TPC_GEMV_LAYOUT
    assert store.geometry is not None and store.geometry["ok"], store.geometry
    assert store.geometry["packed_row"] == hidden // 2
    for kind in store._KINDS:
        packed = getattr(store, f"{kind}_packed")
        scales = getattr(store, f"{kind}_scale")
        scale2 = getattr(store, f"{kind}_scale2")
        assert packed.shape == (experts, rows, hidden // 2)
        assert packed.dtype == torch.uint8
        assert scales.shape == (experts, rows, hidden // 16)
        assert scales.dtype == torch.uint8          # E4M3 bytes, not upcast
        assert scale2.shape == (experts, )
        assert scale2.dtype == torch.float32
        for expert in (0, 7, experts - 1):
            base = f"{CKPT_LM}.layers.0.mlp.experts.{expert}.{kind}"
            assert torch.equal(packed[expert], on_disk[f"{base}.weight"])
            assert torch.equal(scales[expert], on_disk[f"{base}.weight_scale"].view(torch.uint8))
            assert scale2[expert].item() == pytest.approx(on_disk[f"{base}.weight_scale_2"].item())

    # Residency contract: 4-bit nibbles + E4M3 scale bytes + F32 scale2. A
    # BF16 plane here would be the 225 GiB path.
    assert not [b for b in store.buffers() if b.dtype == torch.bfloat16]
    resident = sum(b.numel() for b in store.buffers() if b.numel())
    bf16_bytes = 3 * experts * rows * hidden * 2
    assert resident * 1 < bf16_bytes


def test_tpc_preshuffle_env_switch_reaches_the_store(tmp_path, monkeypatch):
    """``VLLM_QWEN4EXP_TPC_LAYOUT`` is applied once, at load, inside the store."""
    from vllm_gaudi.ops import nvfp4_tpc_layout as tpc_layout

    experts, rows, hidden = 64, 8, 128  # K/2 = 64 bytes -> padded to 256
    root = tmp_path / "padded-ckpt"
    on_disk = _write_expert_group(root, experts=experts, rows=rows, hidden=hidden)
    source = CheckpointTensorSource(root)
    store = NvFp4ExpertStore(experts)

    with monkeypatch.context() as patch:
        patch.setenv(tpc_layout.LAYOUT_ENV, tpc_layout.TPC_GEMV_LAYOUT_PADDED)
        store.load_from_source(_mixed_config(), source, f"{CKPT_LM}.layers.0.mlp")

    assert store.layout == tpc_layout.TPC_GEMV_LAYOUT_PADDED
    packed = store.gate_proj_packed
    assert packed.shape == (experts, rows, 256)
    assert packed.dtype == torch.uint8
    assert int(packed[:, :, hidden // 2:].abs().sum()) == 0, "row padding must be E2M1 0"
    for expert in (0, experts - 1):
        base = f"{CKPT_LM}.layers.0.mlp.experts.{expert}.gate_proj"
        assert torch.equal(tpc_layout.unpreshuffle_packed(packed[expert].unsqueeze(0), hidden)[0],
                           on_disk[f"{base}.weight"])
        assert torch.equal(
            tpc_layout.unpreshuffle_scales(store.gate_proj_scale[expert].unsqueeze(0), hidden)[0],
            on_disk[f"{base}.weight_scale"].view(torch.uint8))
    # A padded plane is not what the Track 3 kernel declares, so the capability
    # check refuses it instead of reading the wrong bytes.
    supported, _ = qwen4_exp.track_c_gemv_capability(
        packed, store.gate_proj_scale, torch.zeros(1, hidden, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.int32))
    assert not supported


def _export_root(which: str) -> Path:
    if which == "plain":
        return Path(os.environ.get("QWEN4EXP_PLAIN_CHECKPOINT",
                                  os.path.expanduser("~/models/qwen4exp-flashnext-plain")))
    override = os.environ.get("QWEN4EXP_CSF_CHECKPOINT")
    if override:
        return Path(override).expanduser()
    hub = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    for candidate in sorted((hub / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local"
                             / "snapshots").glob("*")):
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    return Path("/nonexistent/qwen4exp-csf")


def test_serve_path_decodes_real_csf_experts_end_to_end():
    """One real layer prefix, both exports, through the class the model uses.

    ``NvFp4ExpertStore.load_from_source`` -> ``HPUModelOptMixedPrecisionConfig.
    load_experts`` -> ``modelopt_csf.load_experts``: the CSF export's
    ``nvfp4_csf_*`` tensors must decode to exactly the converted export's plain
    plane, and the store must hold packed U8 nibbles + E4M3 scale bytes.
    """
    roots = {"csf": _export_root("csf"), "plain": _export_root("plain")}
    for name, root in roots.items():
        if not (root / "model.safetensors.index.json").is_file():
            pytest.skip(f"{name} export not available at {root}")

    config = _mixed_config()
    stores: dict[str, NvFp4ExpertStore] = {}
    for name, root in roots.items():
        store = NvFp4ExpertStore(2)
        store.load_from_source(config, CheckpointTensorSource(root), f"{CKPT_LM}.layers.0.mlp")
        stores[name] = store

    for kind in stores["csf"]._KINDS:
        for field in ("packed", "scale", "scale2"):
            from_csf = getattr(stores["csf"], f"{kind}_{field}")
            from_plain = getattr(stores["plain"], f"{kind}_{field}")
            assert torch.equal(from_csf, from_plain), f"{kind}.{field} differs between exports"

    store = stores["csf"]
    from vllm_gaudi.ops import nvfp4_tpc_layout as tpc_layout

    assert store.layout == tpc_layout.TPC_GEMV_LAYOUT
    assert store.gate_proj_packed.dtype == torch.uint8 and store.gate_proj_packed.shape == (2, 640, 1280)
    assert store.gate_proj_scale.dtype == torch.uint8 and store.gate_proj_scale.shape == (2, 640, 160)
    assert store.down_proj_packed.shape == (2, 2560, 320)
    assert store.gate_proj_scale2.dtype == torch.float32
    assert not [buffer for buffer in store.buffers() if buffer.dtype == torch.bfloat16]


def test_streamed_packed_experts_still_load(tmp_path):
    """``--load-format qwen4exp_plain`` compatibility: the streamed pile works.

    ``iter_bf16_weights(packed_experts=True)`` emits the planes as three
    separate tensors; ``PackedRoutedExperts.load_tensor`` is what consumes
    them, so a Track B store must still accept that path unchanged.
    """
    store = NvFp4ExpertStore(2)
    generator = torch.Generator().manual_seed(4)
    weight = torch.randint(0, 256, (8, 64), generator=generator, dtype=torch.uint8)
    scale = torch.randint(60, 63, (8, 8), generator=generator, dtype=torch.uint8)
    assert store.load_tensor("experts.1.gate_proj.weight", weight)
    assert store.load_tensor("experts.1.gate_proj.weight_scale", scale.view(torch.float8_e4m3fn))
    assert store.load_tensor("experts.1.gate_proj.weight_scale_2", torch.tensor([0.125], dtype=torch.float32))
    assert store.load_tensor("experts.1.gate_proj.input_scale", torch.tensor([1.0]))
    assert not store.load_tensor("experts.1.gate_proj.bias", weight)  # not consumed
    assert torch.equal(store.gate_proj_packed[1], weight)
    assert torch.equal(store.gate_proj_scale[1], scale)
    assert store.gate_proj_scale2[1].item() == pytest.approx(0.125)
    assert store.gate_proj_packed.shape == (2, 8, 64)


def test_track_c_operator_path_and_fallback(tmp_path):
    """Track C is used when registered; the fallback stays numerically exact.

    The Track C operator is patched with the exact E2M1 x E4M3 decode (the
    real kernel's contract: F32 accumulate, ``scale2`` after the dot). Both
    paths are compared against the same decode, so the test fails if the store
    passes the wrong planes or folds ``scale2`` twice.
    """
    experts, rows, hidden = 64, 8, 128
    root = tmp_path / "track-c-ckpt"
    write_plans = _write_expert_group(root, experts=experts, rows=rows, hidden=hidden)
    del write_plans
    source = CheckpointTensorSource(root)
    store = NvFp4ExpertStore(experts)
    store.load_from_source(_mixed_config(), source, f"{CKPT_LM}.layers.0.mlp")

    generator = torch.Generator().manual_seed(5)
    ids = torch.tensor([3, 3, 17, 63], dtype=torch.int32)
    x = torch.randn(4, hidden, generator=generator, dtype=torch.bfloat16)

    # The operator is not registered on a host without a device.
    supported, reason = track_c_gemv_capability(store.gate_proj_packed, store.gate_proj_scale, x, ids)
    assert not supported and "not registered" in reason
    # Caps are checked before dispatch (K % 128 / N % 4 / S <= 64 / E range).
    narrow = torch.randn(4, 126, generator=generator, dtype=torch.bfloat16)
    ok, why = track_c_gemv_capability(torch.empty(64, 8, 63, dtype=torch.uint8),
                                      torch.empty(64, 8, 8, dtype=torch.uint8), narrow, ids,
                                      op_available=True)
    assert not ok and "multiple of 128" in why
    ok, why = track_c_gemv_capability(torch.empty(4, 8, 64, dtype=torch.uint8),
                                      torch.empty(4, 8, 8, dtype=torch.uint8), x, ids, op_available=True)
    assert not ok and "E=4" in why

    calls: list[tuple] = []

    def fake_op(ids_, packed, scales, x_, scale2):
        calls.append((tuple(ids_.shape), tuple(packed.shape), tuple(scales.shape), tuple(x_.shape)))
        outs = []
        for slot in range(x_.shape[0]):
            weight = modelopt_csf.dequant_nvfp4(packed[int(ids_[slot])], scales[int(ids_[slot])], 1.0)
            outs.append((x_[slot].float() @ weight.t()) * scale2[int(ids_[slot])])
        return torch.stack(outs).to(torch.bfloat16)

    real_custom_op = qwen4_exp._custom_op
    try:
        qwen4_exp._custom_op = lambda name: fake_op if name == qwen4_exp._TRACK_C_GEMV else None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(qwen4_exp, "track_c_gemv_capability", lambda *a, **k: (True, ""))
            track_c = store.forward_proj("gate_proj", ids, x)
    finally:
        qwen4_exp._custom_op = real_custom_op

    assert calls == [(tuple(ids.shape), (experts, rows, hidden // 2), (experts, rows, hidden // 16),
                      tuple(x.shape))]
    assert store.backends["gate_proj"] == "track_c"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_QWEN4EXP_TRACK_C_MOE", "0")
        dequant = store.forward_proj("gate_proj", ids, x)
    assert store.backends["gate_proj"] == "dequant"

    # Exact reference: float64 decode of the same planes, scale2 after the dot.
    reference = []
    for slot in range(ids.numel()):
        weight = modelopt_csf.dequant_nvfp4(
            store.gate_proj_packed[int(ids[slot])], store.gate_proj_scale[int(ids[slot])], 1.0).double()
        reference.append((x[slot].double() @ weight.t()) * float(store.gate_proj_scale2[int(ids[slot])]))
    reference = torch.stack(reference)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(a.double().reshape(1, -1), b.double().reshape(1, -1)))

    assert cosine(track_c, reference) > 0.99999
    assert cosine(dequant, reference) > 0.9999
    # Same algebra, different rounding: the fallback rounds the decoded weight
    # to BF16 before the matmul, so it may not be elementwise identical.
    assert cosine(track_c, dequant) > 0.9999


# ---------------------------------------------------------------------------
# (4) PLE host offload
# ---------------------------------------------------------------------------
def test_ple_tables_stay_in_host_memory(tmp_path, monkeypatch):
    """The n-gram table is host-resident and only per-token rows are moved."""
    root = tmp_path / "ple-ckpt"
    write_tiny_model(root, quant=False)
    embedding = load_ple_host_embedding(root, layer_idx=1, heads_per_ngram=8, ngram_size=3)
    assert isinstance(embedding, HostPLEEmbedding)
    assert embedding.host_mode == "mmap"
    assert embedding.host_bytes == PLE_TABLE_ROWS * (PLE_HEAD_DIM // 2 + PLE_HEAD_DIM // 16)
    for table in embedding.tables:
        assert table.packed.device.type == "cpu"
        assert table.scales.device.type == "cpu"
        assert table.packed.shape == (PLE_TABLE_ROWS, PLE_HEAD_DIM // 2)

    ids = torch.tensor([[3, 5, 7, 9] * 4], dtype=torch.int64)
    transfers: list[tuple] = []
    real_transfer = qwen4_exp._transfer_host_rows

    def spy(rows, device):
        if device is not None:  # the host-internal calls carry no device
            transfers.append(int(rows.numel()))
        return real_transfer(rows, device)

    monkeypatch.setattr(qwen4_exp, "_transfer_host_rows", spy)
    decoded = embedding.lookup(ids, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert decoded.shape == (1, PLE_HEADS * PLE_HEAD_DIM)
    assert decoded.dtype == torch.bfloat16
    # Only the decoded token rows ever leave the host tables.
    assert transfers and max(transfers) == PLE_HEADS * PLE_HEAD_DIM
    assert sum(transfers) <= 2 * PLE_HEADS * PLE_HEAD_DIM

    # Values match the reference shard decode (mmap must not change the bytes).
    from vllm_gaudi.models.qwen4exp.ple_embedding import PLEShardTable

    host_table = embedding.tables[0]
    packed_rows, scale_rows = host_table.lookup_packed(ids.reshape(-1))
    expected = PLEShardTable.dequant(packed_rows, scale_rows, host_table.global_scale, torch.bfloat16, None)
    torch.testing.assert_close(decoded.reshape(-1), expected.reshape(-1))


def test_ple_offload_switch(monkeypatch):
    """VLLM_PLE_CPU_OFFLOAD: 1 (default) mmap, 0 legacy pinned."""
    monkeypatch.delenv("VLLM_PLE_CPU_OFFLOAD", raising=False)
    assert ple_cpu_offload_enabled() is True
    for value, expected in (("1", True), ("true", True), ("0", False), ("false", False), ("off", False)):
        monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", value)
        assert ple_cpu_offload_enabled() is expected

    # The legacy branch must ask for pinned host copies (and stay off-device).
    calls: list[dict] = []

    def fake_loader(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(host_mode="pinned")

    import vllm_gaudi.models.qwen4exp.ple_runtime as ple_runtime

    monkeypatch.setattr(ple_runtime, "load_ple_ngram", fake_loader)
    result = load_ple_host_embedding(Path("/nonexistent"), layer_idx=1, heads_per_ngram=8, ngram_size=3,
                                     offload=False)
    assert result.host_mode == "pinned"
    assert calls and calls[0]["pin_memory"] is True


# ---------------------------------------------------------------------------
# (5) the model: construction + a real CPU forward with the HPU ops mocked
# ---------------------------------------------------------------------------
def test_dummy_model_forward_on_cpu(tmp_path, fake_gdn, monkeypatch):
    """4-layer forward: shape, finiteness, PLE residency, Track B/C wiring."""
    root = tmp_path / "tiny-model"
    write_tiny_model(root, quant=False)
    model, vllm_config = _build_model(root)
    lm = model.language_model
    layers = list(lm.model.layers)

    # -- geometry ----------------------------------------------------------
    assert len(layers) == 4
    assert sum(1 for layer in layers if hasattr(layer, "linear_attn")) == 3
    assert sum(1 for layer in layers if hasattr(layer, "self_attn")) == 1
    ple_layers = [i for i, layer in enumerate(layers) if layer.ple is not None]
    assert ple_layers == [1]  # ple_layer_ids [2] -> layer_idx + 1
    assert all(isinstance(layer.mlp.experts, NvFp4ExpertStore) for layer in layers)
    assert all(layer.attn_hyper_connection.hc_count == HC for layer in layers)
    # Routed experts are packed buffers, never parameters.
    assert not [name for name, _ in layers[0].mlp.experts.named_parameters()]

    # -- weight-free build has no quantization -----------------------------
    assert lm.quant_config is None
    assert not isinstance(layers[0].mlp.shared_expert.gate_up_proj.quant_method, HPUModelOptMxFp8LinearMethod)

    # -- host PLE tables ---------------------------------------------------
    attached = model.attach_ple_tables(str(root))
    assert attached == 1
    ngram = layers[1].ple.ngram
    assert isinstance(ngram, HostPLEEmbedding) and ngram.host_mode == "mmap"
    table_bytes = ngram.host_bytes
    assert table_bytes == PLE_TABLE_ROWS * (PLE_HEAD_DIM // 2 + PLE_HEAD_DIM // 16)
    # Not a parameter, not a buffer, not a module, not in state_dict: nothing
    # in the module tree can move the table or serialize it.
    from vllm_gaudi.models.qwen4_exp import HostPLETable

    assert not [name for name, _ in model.named_buffers() if "ngram" in name]
    assert not [name for name in model.state_dict() if "ngram" in name]
    assert not [m for m in model.modules() if isinstance(m, HostPLETable)]
    assert not [b for b in model.buffers() if b.numel() == table_bytes]
    for table in ngram.tables:
        assert table.packed.device.type == "cpu" and table.scales.device.type == "cpu"

    # -- experts: small NVFP4 planes so the CPU decode math really runs -----
    fill_expert_planes(model)
    tokens = 5
    ids = torch.randint(0, VOCAB, (tokens, ), generator=torch.Generator().manual_seed(3))
    positions = torch.arange(tokens)

    transfers: list[int] = []
    real_transfer = qwen4_exp._transfer_host_rows

    def spy(rows, device):
        transfers.append(int(rows.numel()))
        return real_transfer(rows, device)

    monkeypatch.setattr(qwen4_exp, "_transfer_host_rows", spy)

    output = model(input_ids=ids, positions=positions)

    assert output.shape == (tokens, HIDDEN)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    # Every PLE lookup transferred at most one token row batch, never a table.
    assert transfers, "PLE never reached the host tables"
    # One decoded token batch per PLE lookup; a table transfer would be
    # PLE_TABLE_ROWS * PLE_HEAD_DIM and a whole table is never moved.
    assert all(n == tokens * PLE_HEADS * PLE_HEAD_DIM for n in transfers), transfers
    assert all(n < PLE_TABLE_ROWS * PLE_HEAD_DIM * 128 for n in transfers)

    store = layers[0].mlp.experts
    assert store.loaded_from is None  # no checkpoint weights were loaded
    assert store.backends  # the CPU fallback recorded itself
    assert set(store.backends.values()) == {"dequant"}
    assert all(layer.mlp.experts.backends["down_proj"] == "dequant" for layer in layers)

    logits = lm.compute_logits(output)
    assert logits.shape == (tokens, VOCAB)
    assert torch.isfinite(logits).all()
    # Hyper-connections collapse the 4 streams before each block, so the
    # mocked GDN layer is fed the single-stream width.
    assert fake_gdn.calls and set(fake_gdn.calls) == {HIDDEN}


def test_dummy_model_forward_uses_track_c_when_available(tmp_path, fake_gdn, monkeypatch):
    """The Track C operator drives the decode path end to end when present."""
    root = tmp_path / "tiny-track-c"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    model.attach_ple_tables(str(root))
    # 64 experts so the kernel caps (E % 64, E >= 64) are satisfied.
    fill_expert_planes(model, experts=64)

    calls: list[tuple] = []

    def fake_op(ids_, packed, scales, x_, scale2):
        calls.append((int(ids_.numel()), int(packed.shape[0]), int(x_.shape[1])))
        outs = []
        for slot in range(x_.shape[0]):
            weight = modelopt_csf.dequant_nvfp4(packed[int(ids_[slot])], scales[int(ids_[slot])], 1.0)
            outs.append((x_[slot].float() @ weight.t()) * scale2[int(ids_[slot])])
        return torch.stack(outs).to(torch.bfloat16)

    tokens = 4
    ids = torch.randint(0, VOCAB, (tokens, ), generator=torch.Generator().manual_seed(9))
    positions = torch.arange(tokens)

    real_custom_op = qwen4_exp._custom_op
    try:
        qwen4_exp._custom_op = lambda name: fake_op if name == qwen4_exp._TRACK_C_GEMV else None
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(qwen4_exp, "track_c_gemv_capability", lambda *a, **k: (True, ""))
            track_c = model(input_ids=ids, positions=positions)
    finally:
        qwen4_exp._custom_op = real_custom_op

    assert torch.isfinite(track_c).all()
    assert track_c.shape == (tokens, HIDDEN)
    assert calls, "Track C operator was not called"
    assert calls[0][1] == 64  # the full expert group is passed, packed
    store = model.language_model.model.layers[0].mlp.experts
    assert store.backends["gate_proj"] == "track_c"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_QWEN4EXP_TRACK_C_MOE", "0")
        fallback = model(input_ids=ids, positions=positions)
    assert store.backends["gate_proj"] == "dequant"
    # Both paths decode the same packed planes; only the weight rounding
    # differs (Track C: F32 decode + F32 dot, fallback: BF16 weight + BF16
    # matmul), so the outputs must agree closely, not bit-for-bit.
    cosine = torch.nn.functional.cosine_similarity(track_c.float().reshape(1, -1),
                                                   fallback.float().reshape(1, -1))
    assert float(cosine.detach()) > 0.99


def test_moe_decode_dispatches_to_h10_h15_when_track_c_is_off(tmp_path, fake_gdn, monkeypatch):
    """Decode takes the H10+H15 fused chain unless a Track C op is registered.

    The production default has no Track C library loaded, so ``_grouped_swiglu``
    must hand the un-replicated ``hidden`` ``[T, H]`` to ``grouped_gemv``
    (gate/up index token rows themselves, one fused SwiGLU GEMV, fused down
    epilogue). Registering a Track C op must restore the per-projection
    contract *including* the ``[T*top_k, H]`` replication, which is what the
    Track C kernels and their tests are written against.
    """
    root = tmp_path / "decode-dispatch"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    fill_expert_planes(model)
    block = model.language_model.model.layers[0].mlp
    store = block.experts

    fused: list[tuple] = []

    def fake_fused(hidden, ids, weights, experts):
        fused.append((tuple(hidden.shape), tuple(ids.shape), tuple(weights.shape)))
        return torch.zeros(hidden.shape, dtype=hidden.dtype)

    monkeypatch.setattr(qwen4_exp, "grouped_gemv", fake_fused)
    monkeypatch.setattr(qwen4_exp, "_fused_decode_gemv", lambda tensor: True)
    assert qwen4_exp.track_c_enabled() is False  # nothing registered on a host

    tokens = 3
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16)
    out = block(hidden)
    assert fused == [((tokens, HIDDEN), (tokens, TOP_K), (tokens, TOP_K))]
    assert out.shape == (tokens, HIDDEN)
    # The fused chain replaced the per-projection loop: it records itself, and
    # no per-projection backend was chosen.
    assert store.backends == {"decode": "h10_h15"}

    # Negative control: a registered Track C operator owns decode again.
    track_c_rows: list[int] = []

    def fake_op(ids_, packed, scales, x_, scale2):
        track_c_rows.append(int(x_.shape[0]))
        outs = []
        for slot in range(x_.shape[0]):
            weight = modelopt_csf.dequant_nvfp4(packed[int(ids_[slot])], scales[int(ids_[slot])], 1.0)
            outs.append((x_[slot].float() @ weight.t()) * scale2[int(ids_[slot])])
        return torch.stack(outs).to(torch.bfloat16)

    fused.clear()
    real_custom_op = qwen4_exp._custom_op
    try:
        qwen4_exp._custom_op = lambda name: fake_op if name == qwen4_exp._TRACK_C_GEMV else None
        assert qwen4_exp.track_c_enabled() is True
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(qwen4_exp, "track_c_gemv_capability", lambda *a, **k: (True, ""))
            track_c = block(hidden)
    finally:
        qwen4_exp._custom_op = real_custom_op

    assert not fused, "Track C must keep its own per-projection path"
    assert store.backends["gate_proj"] == "track_c"
    assert track_c_rows and set(track_c_rows) == {tokens * TOP_K}  # slots, not tokens
    assert track_c.shape == (tokens, HIDDEN)


def test_shared_expert_h15_resolves_both_gate_spellings(tmp_path, fake_gdn):
    """The fused shared expert finds the scalar gate on the MLP *or* the block.

    ``Qwen2MoeMLP`` -- imported as ``Qwen3NextMLP`` in ``qwen4_exp.py`` -- is
    built with ``expert_gate=`` here, so the gate sits on the MLP; the other
    spelling keeps ``shared_expert_gate`` on the parent block. Both must
    resolve to the same three weights, and anything uncovered (no gate, a
    bias, a width mismatch) must decline so the stock MLP runs.
    """
    root = tmp_path / "shared-gate"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    block = model.language_model.model.layers[0].mlp
    mlp = block.shared_expert

    parts = shared_expert_h15_parts(mlp)
    assert parts is not None
    gate_up, down, gate_w, down_w, shared_w = parts
    assert gate_up is mlp.gate_up_proj and down is mlp.down_proj
    assert gate_w is mlp.gate_up_proj.weight
    assert shared_w is block.shared_expert_gate.weight  # carried on the MLP too
    assert gate_w.shape == (2 * MOE_INTER, HIDDEN)
    assert down_w.shape == (HIDDEN, MOE_INTER)
    assert shared_w.shape == (1, HIDDEN)

    class _MlpWithoutGate(torch.nn.Module):
        """Block-style shared expert: the scalar gate lives beside the MLP."""

        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = mlp.gate_up_proj
            self.down_proj = mlp.down_proj

    parts = shared_expert_h15_parts(_MlpWithoutGate(), block.shared_expert_gate)
    assert parts is not None and parts[4] is block.shared_expert_gate.weight
    assert shared_expert_h15_parts(_MlpWithoutGate()) is None  # no gate anywhere

    # The MLP's own gate wins when both spellings are present.
    decoy = torch.nn.Linear(HIDDEN, 1, bias=False).to(torch.bfloat16)
    assert shared_expert_h15_parts(mlp, decoy)[4] is mlp.expert_gate.weight

    # Uncovered shapes decline instead of guessing.
    class _Biased(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Linear(HIDDEN, 2 * MOE_INTER, bias=True).to(torch.bfloat16)
            self.down_proj = torch.nn.Linear(MOE_INTER, HIDDEN, bias=False).to(torch.bfloat16)

    assert shared_expert_h15_parts(_Biased(), block.shared_expert_gate) is None
    # Gate width must match the MLP hidden width.
    narrow_mlp = SimpleNamespace(
        gate_up_proj=torch.nn.Linear(MOE_INTER, 2 * MOE_INTER, bias=False).to(torch.bfloat16),
        down_proj=torch.nn.Linear(MOE_INTER, MOE_INTER, bias=False).to(torch.bfloat16),
    )
    assert narrow_mlp.gate_up_proj.weight.shape == (2 * MOE_INTER, MOE_INTER)
    assert shared_expert_h15_parts(narrow_mlp, block.shared_expert_gate) is None
    # Host tensors never take the fused path, whatever the weights look like.
    assert shared_expert_forward_h15(mlp, torch.randn(2, HIDDEN, dtype=torch.bfloat16)) is None


def test_shared_expert_h15_declines_track_b_fp8_weights(tmp_path, fake_gdn):
    """A Track B (MXFP8) shared expert keeps the stock MLP, never the fusion."""
    root = tmp_path / "shared-quant"
    write_tiny_model(root, quant=True)
    model, _ = _build_model(root)
    block = model.language_model.model.layers[0].mlp
    shared = block.shared_expert

    assert shared.gate_up_proj.weight.dtype == torch.float8_e4m3fn
    assert isinstance(shared.gate_up_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert shared.expert_gate is block.shared_expert_gate  # the gate itself stays BF16
    assert shared_expert_h15_parts(shared) is None


def test_mixed_precision_model_dispatches_per_prefix(tmp_path, fake_gdn):
    """A checkpoint declaring MIXED_PRECISION builds Track B layers."""
    root = tmp_path / "tiny-quant"
    write_tiny_model(root, quant=True)
    model, vllm_config = _build_model(root)
    lm = model.language_model

    assert isinstance(lm.quant_config, HPUModelOptMixedPrecisionConfig)
    # Installed on the config so the nested HPU GDN layer sees it too.
    assert vllm_config.quant_config is lm.quant_config
    assert lm.quant_config.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]

    layers = list(lm.model.layers)
    shared = layers[0].mlp.shared_expert
    assert isinstance(shared.gate_up_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert isinstance(shared.down_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert shared.gate_up_proj.weight.dtype == torch.float8_e4m3fn

    qsa = layers[3].self_attn
    assert isinstance(qsa.qkv_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert isinstance(qsa.o_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert isinstance(qsa.indexer.index_qk_proj.quant_method, HPUModelOptMxFp8LinearMethod)

    # Not declared -> stays BF16 (hyper-connections are never quantized).
    assert type(layers[0].attn_hyper_connection.input_mix_weight_down_block_inject
                .quant_method).__name__ == "UnquantizedLinearMethod"
    # The checkpoint's ignore list keeps the head in BF16.
    assert type(lm.lm_head.quant_method).__name__ == "UnquantizedLinearMethod"
    # The routed expert store is filled by Track B, not by a dense linear.
    assert isinstance(layers[0].mlp.experts, NvFp4ExpertStore)
    assert lm._experts_from_source is True
    assert lm._source is not None


# ---------------------------------------------------------------------------
# (6) GDN identity: the plugin class itself, not a mock
# ---------------------------------------------------------------------------
def test_gdn_linear_attention_is_the_plugin_class_without_a_mock(tmp_path):
    """The production decoder builds the real HPU GatedDeltaNet, unpatched.

    ``resolve_gdn_backend`` reports the class the decoder's module-attribute
    lookup resolves to, and the op modules that class is built from. Building
    the layer for real (no stub) therefore proves the identity; only its
    *forward* needs the stub offline, because every GDN kernel is HPU-only
    (no CPU fallback) and the on-card smoke runs it for real.
    """
    import vllm_gaudi.models.qwen3_5 as qwen3_5

    root = tmp_path / "tiny-gdn"
    write_tiny_model(root, quant=True)  # declares the GDN projections MXFP8
    model, _ = _build_model(root)
    layers = list(model.language_model.model.layers)

    gdn_layers = [layer for layer in layers if hasattr(layer, "linear_attn")]
    assert len(gdn_layers) == 3
    for layer in gdn_layers:
        assert type(layer.linear_attn) is qwen3_5.HPUGatedDeltaNetAttention
        assert type(layer.linear_attn).__module__ == "vllm_gaudi.models.qwen3_5"
    # The model resolved the class before building the layers and verified the
    # built class: the reported identity and the installed one cannot drift.
    assert model.language_model._gdn_layers_built == 3
    assert model.language_model._gdn_backend.identity == (
        f"{qwen3_5.HPUGatedDeltaNetAttention.__module__}."
        f"{qwen3_5.HPUGatedDeltaNetAttention.__name__}")

    backend = resolve_gdn_backend()
    assert backend.is_plugin is True
    assert backend.module == "vllm_gaudi.models.qwen3_5"
    assert backend.class_name == "HPUGatedDeltaNetAttention"
    assert backend.ops == {
        "flashinfer_gaudi.gdn_decode": "loaded",
        "vllm_gaudi.ops.hpu_gdn_pytorch": "loaded",
        "vllm_gaudi.ops.causal_conv1d_pytorch": "loaded",
        "vllm_gaudi.models.qwen3_5": "loaded",
    }
    # The class body really calls those ops (not just imports them).
    source = inspect.getsource(qwen3_5.HPUGatedDeltaNetAttention)
    for symbol in ("maybe_run_gdn_prefill", "maybe_run_gdn_decode_packed",
                   "hpu_causal_conv1d_fn", "hpu_causal_conv1d_update"):
        assert symbol in source, symbol

    # The Qwen4Exp GDN geometry is the plugin path's contract, unchanged.
    gdn = gdn_layers[0].linear_attn
    assert (gdn.num_k_heads, gdn.num_v_heads) == (TEXT_CONFIG["linear_num_key_heads"],
                                                   TEXT_CONFIG["linear_num_value_heads"])
    assert (gdn.head_k_dim, gdn.head_v_dim) == (TEXT_CONFIG["linear_key_head_dim"],
                                                TEXT_CONFIG["linear_value_head_dim"])
    assert gdn.conv_kernel_size == TEXT_CONFIG["linear_conv_kernel_dim"]

    # Track 4.5: the MXFP8 declarations reach the plugin GDN's own projections.
    assert isinstance(gdn.in_proj_qkvz.quant_method, HPUModelOptMxFp8LinearMethod)
    assert isinstance(gdn.in_proj_ba.quant_method, HPUModelOptMxFp8LinearMethod)
    assert isinstance(gdn.out_proj.quant_method, HPUModelOptMxFp8LinearMethod)
    assert gdn.in_proj_qkvz.weight.dtype == torch.float8_e4m3fn
    assert gdn.in_proj_qkvz.weight_scale.dtype == torch.uint8  # one UE8M0 byte / 32 inputs


# ---------------------------------------------------------------------------
# (7) QSA: exact fp32 for short steps, BF16 for prefill buckets
# ---------------------------------------------------------------------------
def test_qsa_routes_exact_for_short_steps_and_bf16_for_prefill(monkeypatch):
    """The routing rule is a pure function, so the numerics choice is testable.

    H23-1 changed the HPU default for the *dense* prefix path (every decode
    step and short prompt, ``S_kv <= indexer_budget``): it now takes the BF16
    MME product, because Gaudi2's MME accumulates a BF16 GEMM in fp32 in
    hardware and the fp32 planes only bought the ``k``/``v`` copy of the whole
    cache per token (measured 1.56x at decode T=1, ``S_kv``=257, in
    ``tools/qwen4exp_flashnext/probe_h23_ops.py``).

    H28-1 removed the reason the gathered sparse path kept the pre-H23 plane
    rule: that rule existed because the fast variant re-gathered the selected
    rows once per 1024-column tile, so a decode call split a 2 MiB gather into
    ``1024 + 1024 + 3`` (measured 2.2x slower at ``S_kv``=2560). The tile is
    now byte-budgeted rather than column-capped, every decode shape is one
    gather, and the sparse path takes the BF16 product too. The plane rule
    survives only as the ``H23`` off / ``H28_SPARSE`` off fallback that an A/B
    run needs to reproduce the old numbers.

    Non-HPU tensors (the offline reference path) always run exact, and
    ``VLLM_QWEN4EXP_QSA_EXACT`` still overrides both ways.
    """
    path = qsa_math.qsa_attention_path
    budget = qsa_math._EXACT_PLANE_BUDGET
    # The checkpoint's QSA geometry: 2 KV heads of head_dim 256 and the fixed
    # indexer width indexer_budget + ratio - 1 = 2051.
    width, kv_heads, head_dim = 2051, 2, 256
    # Dense prefix path (H23-1): fast on HPU whatever the plane size.
    assert path(1, width, kv_heads, head_dim, device_type="hpu", exact_pref="",
                dense=True) == "fast"
    assert path(60, width, kv_heads, head_dim, device_type="hpu", exact_pref="",
                dense=True) == "fast"
    # Gathered sparse path (H28-1): fast on HPU at every shape.
    # Decode: 1 x 2051 x 2 x 256 x 2 B = 2.0 MiB, one untiled gather.
    assert path(1, width, kv_heads, head_dim, device_type="hpu", exact_pref="") == "fast"
    assert path(60, width, kv_heads, head_dim, device_type="hpu", exact_pref="") == "fast"
    # The 2048-token prefill bucket is 8.0 GiB fp32 -> BF16 path as before.
    assert path(2048, width, kv_heads, head_dim, device_type="hpu", exact_pref="") == "fast"
    # The untiled rule is what makes decode fast: the byte budget must cover
    # the fixed width for every decode-shaped query count.
    for queries in (1, 4, 8, 32, 64, 128, 256):
        assert qsa_math._fast_width_tile(queries, kv_heads, head_dim) >= width, queries
    # Prefill buckets still tile (a 2048-token one at 512 columns, i.e. the
    # pre-H28 tile width and the same numerics).
    assert qsa_math._fast_width_tile(2048, kv_heads, head_dim) == 512
    assert qsa_math._fast_width_tile(1024, kv_heads, head_dim) == 1024
    # CPU (the offline reference path) is always exact, whatever the size.
    assert path(2048, width, kv_heads, head_dim, device_type="cpu", exact_pref="0") == "exact"
    assert path(1, width, kv_heads, head_dim, device_type="cpu", exact_pref="",
                dense=True) == "exact"
    # Env overrides.
    assert path(2048, width, kv_heads, head_dim, device_type="hpu", exact_pref="1") == "exact"
    assert path(1, width, kv_heads, head_dim, device_type="hpu", exact_pref="0") == "fast"
    assert path(1, width, kv_heads, head_dim, device_type="hpu", exact_pref="1",
                dense=True) == "exact"
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_EXACT", "1")
    assert path(2048, width, kv_heads, head_dim, device_type="hpu") == "exact"
    assert path(1, width, kv_heads, head_dim, device_type="hpu", dense=True) == "exact"
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_EXACT", "0")
    assert path(1, width, kv_heads, head_dim, device_type="hpu") == "fast"
    monkeypatch.delenv("VLLM_QWEN4EXP_QSA_EXACT")
    # H28-1 kill switch: the sparse route falls back to the fp32 plane rule.
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H28_SPARSE", "0")
    assert path(1, width, kv_heads, head_dim, device_type="hpu") == "exact"
    # 63 queries are 264.6 MB, 64 are 268.8 MB against the 268.4 MB budget.
    per_query = width * kv_heads * head_dim * 4
    at = budget // per_query
    assert at == 63 and at * per_query <= budget < (at + 1) * per_query
    assert path(at, width, kv_heads, head_dim, device_type="hpu") == "exact"
    assert path(at + 1, width, kv_heads, head_dim, device_type="hpu") == "fast"
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H28_SPARSE", "1")
    # H23 kill switch: both HPU routes fall back to the pre-H23 plane rule.
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H23", "0")
    assert path(1, width, kv_heads, head_dim, device_type="hpu", dense=True) == "exact"
    assert path(2048, width, kv_heads, head_dim, device_type="hpu", dense=True) == "fast"
    assert path(1, width, kv_heads, head_dim, device_type="hpu") == "exact"
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H23", "1")
    assert path(1, width, kv_heads, head_dim, device_type="hpu", dense=True) == "fast"


def test_qsa_h23_selection_rewrites_are_index_identical(monkeypatch):
    """H23-3's two rewrites must not move a single selected index.

    * descending stable ``argsort`` instead of ``argsort(-scores)``;
    * the "every pooled block is visible" shortcut that skips the ``-inf``
      mask, the gather and the count on a decode step.

    Both are exercised on a real selection (decode rows at several positions,
    including a query that does not yet see every block, plus relu-heavy
    score ties) and compared against the pre-H23 code with the switches off.
    """
    torch.manual_seed(19)
    budget, ratio, heads, dim = 64, 4, 2, 16
    total_blocks = 40
    pooled = torch.randn(total_blocks, dim)
    q = torch.randn(3, heads, dim)
    positions = torch.tensor(
        [total_blocks * ratio - 1,          # sees every block
         total_blocks * ratio - 5,          # misses the last block
         total_blocks * ratio // 2])        # mid history
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H23", "0")
    before = qsa_math.select_qsa_indices_from_pooled(
        q, pooled, positions, total_blocks=total_blocks,
        token_budget=budget, compress_ratio=ratio,
        total_tokens=total_blocks * ratio)
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_H23", "1")
    after = qsa_math.select_qsa_indices_from_pooled(
        q, pooled, positions, total_blocks=total_blocks,
        token_budget=budget, compress_ratio=ratio,
        total_tokens=total_blocks * ratio)
    # The host-side "all visible" proof is what the layer passes on a decode
    # step: the smallest logical position of the committed chunk.
    host_min = int(positions.min())
    shortcut = qsa_math.select_qsa_indices_from_pooled(
        q, pooled, positions, total_blocks=total_blocks,
        token_budget=budget, compress_ratio=ratio,
        total_tokens=total_blocks * ratio, host_min_position=host_min)
    assert torch.equal(before, after), "H23-3 rewrites changed the selection"
    assert torch.equal(before, shortcut), "all-visible shortcut changed the selection"
    # And the descending stable sort really is the negated-sort order.
    scores = torch.relu(torch.randn(1, 256)).round(decimals=2)
    assert torch.equal(torch.argsort(-scores, dim=-1, stable=True),
                       torch.argsort(scores, dim=-1, stable=True, descending=True))
    # A single-row decode step that sees every block selects k * ratio tokens.
    single = qsa_math.select_qsa_indices_from_pooled(
        q[:1], pooled, positions[:1], total_blocks=total_blocks,
        token_budget=budget, compress_ratio=ratio,
        total_tokens=total_blocks * ratio, host_min_position=total_blocks * ratio - 1)
    assert int((single[0] >= 0).sum()) == min(budget, total_blocks * ratio)


def _qsa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Straight-loop GQA over ``indices`` in float64, mask by ``-1`` pad."""
    num_queries, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    n_rep = num_q_heads // num_kv_heads
    out = torch.zeros(num_queries, num_q_heads, head_dim, dtype=torch.float64)
    for t in range(num_queries):
        for head in range(num_q_heads):
            kv_head = head // n_rep
            picked = [int(i) for i in indices[t] if int(i) >= 0]
            if not picked:
                continue
            dots = torch.stack([(q[t, head].double() * k[i, kv_head].double()).sum() for i in picked])
            weights = torch.softmax(dots * scale, dim=-1)
            out[t, head] = sum(w * v[i, kv_head].double() for w, i in zip(weights, picked))
    return out


def test_qsa_exact_and_fast_paths_agree_and_match_a_loop_reference():
    """Both paths implement the same attention; the fast one only rounds to BF16.

    A loop reference (float64) pins the numerics. The exact path must match it
    tightly, the tiled exact path exactly, and the BF16 path within one BF16
    rounding — that is the prefill-bucket compromise, not a different formula.
    """
    torch.manual_seed(7)
    q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(7, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(7, 2, 8, dtype=torch.bfloat16)
    indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [0, 2, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    scale = 8**-0.5
    reference = _qsa_reference(q, k, v, indices, scale)

    # CPU tensors always take the exact path.
    exact = qsa_math.sparse_attention(q, k, v, indices, scale)
    tiled = qsa_math._sparse_attention_width_tiled(q, k, v, indices, scale)
    fast = qsa_math._sparse_attention_fast(q, k, v, indices, scale)

    for name, out in (("exact", exact), ("tiled", tiled)):
        assert out.dtype == torch.bfloat16 and out.shape == (3, 4, 8)
        torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=2e-2,
                                   msg=lambda m, n=name: f"{n}: {m}")
    torch.testing.assert_close(tiled, exact, rtol=0, atol=0)
    # Row with no selected token is identically zero in every path.
    for out in (exact, tiled, fast):
        assert torch.equal(out[2], torch.zeros_like(out[2]))
    # Fast path: same algebra, BF16-rounded GEMM inputs.
    torch.testing.assert_close(fast.double(), reference, rtol=5e-2, atol=5e-2)
    assert float(torch.nn.functional.cosine_similarity(
        fast.double().reshape(1, -1), reference.reshape(1, -1))) > 0.9999


def test_qsa_fast_attention_single_shot_matches_the_tiled_loop(monkeypatch):
    """H28-1: the untiled fast path is the same algebra as the tiled one.

    ``_sparse_attention_fast`` now takes one gather when the byte budget covers
    the whole selection (every decode shape) and falls back to key-column tiles
    otherwise. Both spellings must land on the loop reference, and the shared
    clamped index tensor must not change which rows are read: the only
    difference between them is where the BF16 rounding of the value product
    happens (one GEMM vs one per tile).
    """
    torch.manual_seed(13)
    q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(9, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(9, 2, 8, dtype=torch.bfloat16)
    indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 5, -1, -1, -1, -1, -1, -1, -1],
         [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    scale = 8**-0.5
    reference = _qsa_reference(q, k, v, indices, scale)
    single = qsa_math._sparse_attention_fast(q, k, v, indices, scale)
    assert qsa_math._fast_width_tile(3, 2, 8) >= indices.shape[-1]
    # One column per tile exercises the tiled loop (and its fp32 accumulator).
    monkeypatch.setattr(qsa_math, "_FAST_TILE_BYTES", 1)
    assert qsa_math._fast_width_tile(3, 2, 8) == 1
    tiled = qsa_math._sparse_attention_fast(q, k, v, indices, scale)
    for name, out in (("single", single), ("tiled", tiled)):
        assert out.dtype == torch.bfloat16 and out.shape == (3, 4, 8)
        torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=2e-2,
                                   msg=lambda m, n=name: f"{n}: {m}")
        # The no-selected-token row is exactly zero in both spellings.
        assert torch.equal(out[2], torch.zeros_like(out[2])), name
    torch.testing.assert_close(single, tiled, rtol=0, atol=2e-2)


def test_qsa_index_selection_is_dense_below_budget_and_capped_above():
    """``select_qsa_indices`` keeps every causal token under the budget."""
    torch.manual_seed(11)
    budget, ratio, head_dim = 8, 2, 8
    q = torch.randn(2, 2, head_dim)
    raw_keys = torch.randn(12, head_dim)
    weight = torch.ones(head_dim)
    positions = torch.tensor([2, 3])
    # block_cos_sin that makes the block RoPE the identity (cos=1, sin=0).
    cos_sin = torch.cat([torch.ones(6, head_dim // 2), torch.zeros(6, head_dim // 2)], dim=-1)

    short = qsa_math.select_qsa_indices(
        q, raw_keys[:4], positions, k_norm_weight=weight, eps=1e-6,
        block_cos_sin=cos_sin[:2], token_budget=budget, compress_ratio=ratio)
    assert short.shape == (2, budget + ratio - 1)
    for row, visible in enumerate((3, 4)):
        picked = sorted(int(i) for i in short[row] if int(i) >= 0)
        assert picked == list(range(visible))  # dense causal: budget not reached

    # Longer history than the budget: at most block_topk blocks + the tail,
    # and the selected tokens are a subset of the causal window. Position 10
    # (11 visible tokens) leaves an incomplete tail block; position 11 (12
    # visible, a multiple of the ratio) has none.
    long_positions = torch.tensor([10, 11])
    long = qsa_math.select_qsa_indices(
        q, raw_keys, long_positions, k_norm_weight=weight, eps=1e-6,
        block_cos_sin=cos_sin, token_budget=4, compress_ratio=ratio)
    assert long.shape == (2, 4 + ratio - 1)
    for row in range(2):
        picked = [int(i) for i in long[row] if int(i) >= 0]
        assert len(picked) <= 4 + ratio - 1
        assert all(0 <= i <= 11 for i in picked)
        assert len(set(picked)) == len(picked)
        assert len(picked) >= 4  # the budget is spent before the tail is added
    assert 10 in [int(i) for i in long[0] if int(i) >= 0]  # incomplete tail kept

    # Windowed selection over the fully pooled history is bit-identical to
    # the full-history selection (the compile-stable bundle width changes no
    # selected index).
    pooled = qsa_math.pool_indexer_blocks(
        raw_keys.view(6, ratio, head_dim), k_norm_weight=weight, eps=1e-6,
        block_cos_sin=cos_sin)
    windowed = qsa_math.select_qsa_indices_from_pooled(
        q, pooled, long_positions, total_blocks=6, token_budget=4, compress_ratio=ratio)
    full = qsa_math.select_qsa_indices(
        q, raw_keys, long_positions, k_norm_weight=weight, eps=1e-6,
        block_cos_sin=cos_sin, token_budget=4, compress_ratio=ratio)
    assert torch.equal(windowed, full)


def test_qsa_software_cache_indexes_and_pools_per_request(tmp_path, fake_gdn, monkeypatch):
    """The per-slot software cache grows incrementally and stays per request."""
    # Pool on every commit (the pre-H30 schedule) so the incremental pooling itself is exercised
    # below the token budget; test_qsa_lazy_pool_matches_eager_pool covers the default.
    monkeypatch.setenv("VLLM_QWEN4EXP_QSA_LAZY_POOL", "0")
    root = tmp_path / "tiny-qsa"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    qsa = model.language_model.model.layers[3].self_attn
    indexer = qsa.indexer
    qsa.reset_bringup_cache()

    hidden = torch.randn(6, HIDDEN, dtype=torch.bfloat16)
    # Two requests of 3 tokens each, both starting at position 0.
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    out = qsa(hidden_states=hidden, positions=positions,
              query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32))
    assert out.shape == (6, HIDDEN) and torch.isfinite(out).all()
    assert len(qsa._bringup) == 2
    for slot in (0, 1):
        state = qsa._bringup[slot]
        assert state.raw_keys.shape == (3, indexer.index_head_dim)
        assert state.key.shape == (3, qsa.num_kv_heads, qsa.head_dim)
        assert state.pooled.shape == (3 // indexer.compress_ratio, indexer.index_head_dim)
    pooled_before = qsa._bringup[0].pooled.clone()
    # Not a buffer: the cache is host-side state, like the PLE conv history.
    assert not [name for name, _ in qsa.named_buffers() if "bringup" in name]
    assert not [name for name in qsa.state_dict() if "bringup" in name]

    # One decode step for request 0 only: slot 0 extends, slot 1 is untouched.
    decode = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    qsa(hidden_states=decode, positions=torch.tensor([3]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32))
    slot0, slot1 = qsa._bringup
    assert slot0.raw_keys.shape[0] == 4
    assert slot1.raw_keys.shape[0] == 3
    # Blocks are pooled once and never recomputed: the old pooled rows are
    # bit-identical after the extension.
    assert torch.equal(slot0.pooled[: pooled_before.shape[0]], pooled_before)
    assert slot0.pooled.shape[0] == 4 // indexer.compress_ratio
    # The indexer's fixed-width output covers budget + ratio - 1 tokens.
    assert indexer.output_width == indexer.token_topk + indexer.compress_ratio - 1

    # A cold start at a position that does not continue the cache rebuilds
    # the slot instead of silently appending unseen history (the runner's
    # warmup dummies carry such a synthetic context; 6144 is outside the tiny
    # model's rotary range, so use a real chunk-start gap).
    qsa(hidden_states=torch.randn(2, HIDDEN, dtype=torch.bfloat16),
        positions=torch.tensor([6, 7]),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32))
    assert qsa._bringup[0].raw_keys.shape[0] == 2


# ---------------------------------------------------------------------------
# (8) Hyper-connections: mix/combine math and the state carried per request
# ---------------------------------------------------------------------------
def _hc_reference(
    x: torch.Tensor,
    module: GatedResidual,
    *,
    block_output: torch.Tensor | None = None,
    injection: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Closed-form ``mix`` / ``combine_and_mix`` written out independently."""
    hc = module.hc_count
    hidden = module.hidden_size
    if block_output is not None and injection is not None:
        stream = x.float().unflatten(-1, (hc, hidden))
        gate = 2.0 * torch.sigmoid(injection.float() / hc)
        stream = stream + block_output.float().unsqueeze(-2) * gate.unsqueeze(-1)
        x = stream.flatten(-2).to(x.dtype)
    grouped = x.float().unflatten(-1, (hc, hidden))
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    weight = module.hc_norm.weight.float().unflatten(-1, (hc, hidden))
    xn = (normalized * (1.0 + weight)).flatten(-2).to(x.dtype)
    merged = module.input_mix_weight_down_block_inject(xn)
    lowrank, new_injection, _ = merged.split([module.hc_lowrank, hc, module.pad_size], dim=-1)
    lowrank = torch.nn.functional.silu(lowrank.float() / hc).to(x.dtype)
    logits = module.input_mix_weight_up(lowrank)
    g = torch.sigmoid(logits.float().unflatten(-1, (hc, hidden)))
    block_input = (g * xn.float().unflatten(-1, (hc, hidden))).mean(dim=-2).to(x.dtype)
    return x, block_input, new_injection


def test_hyper_connection_mix_and_combine_match_the_closed_form():
    """hc_count=4 mixing/injection math equals its definition, row for row."""
    torch.manual_seed(21)
    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=8,
        params_dtype=torch.bfloat16,
        hc_lowrank=4,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    module = GatedResidual(config, prefix="probe")
    assert module.hc_count == 4 and module.pad_size == (-(4 + 4)) % 16 == 8
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.05)

    x = torch.randn(5, HC * 8, dtype=torch.bfloat16)
    hidden, block_input, injection = module.mix(x)
    _, ref_block_input, ref_injection = _hc_reference(x, module)

    # mix() is a pure function of the current row: the residual stream is
    # returned unchanged and the block input/injection are per-row.
    assert torch.equal(hidden, x)
    torch.testing.assert_close(block_input, ref_block_input, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(injection, ref_injection, rtol=0, atol=0)
    assert block_input.shape == (5, 8) and injection.shape == (5, HC)

    # combine(): residual + block_output * (2 * sigmoid(inj / hc)) per stream.
    block_output = torch.randn(5, 8, dtype=torch.bfloat16)
    combined = module.combine(x, block_output, injection)
    streams = x.float().unflatten(-1, (HC, 8))
    expected = (streams
                + block_output.float().unsqueeze(-2)
                * (2.0 * torch.sigmoid(injection.float() / HC)).unsqueeze(-1)).flatten(-2)
    # combine() rounds once, to the residual's dtype, like the reference.
    torch.testing.assert_close(combined, expected.to(combined.dtype), rtol=0, atol=0)

    # combine_and_mix() fuses the two: same boundary as combine() then mix().
    fused_hidden, fused_input, fused_injection = module.combine_and_mix(x, block_output, injection)
    torch.testing.assert_close(fused_hidden, combined, rtol=0, atol=0)
    _, after_combine_input, after_combine_injection = module.mix(combined)
    torch.testing.assert_close(fused_input, after_combine_input, rtol=0, atol=0)
    torch.testing.assert_close(fused_injection, after_combine_injection, rtol=0, atol=0)


def test_hyper_connection_state_is_row_local_across_requests():
    """The pending HC state is per token row, so requests never mix.

    Two requests batched as rows 0-2 and 3-4 must produce exactly what each
    one produces alone: the (block_output, injection) tuple carried between
    layers is row-indexed, not a shared accumulator.
    """
    torch.manual_seed(22)
    config = HyperConnectionConfig(
        hc_count=HC, hidden_size=8, params_dtype=torch.bfloat16,
        hc_lowrank=4, rms_norm_eps=1e-6, hc_per_branch_norm=True,
    )
    module = GatedResidual(config, prefix="probe")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.05)
    x = torch.randn(5, HC * 8, dtype=torch.bfloat16)
    block_output = torch.randn(5, 8, dtype=torch.bfloat16)
    injection = torch.randn(5, HC, dtype=torch.bfloat16)

    batched = module.combine_and_mix(x, block_output, injection)
    for rows in (slice(0, 3), slice(3, 5)):
        alone = module.combine_and_mix(x[rows], block_output[rows], injection[rows])
        for got, want in zip((batched[0][rows], batched[1][rows], batched[2][rows]), alone):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    # The small helper functions agree with the module-level path.
    torch.testing.assert_close(
        hc_combine(x, block_output, injection, HC), module.combine(x, block_output, injection),
        rtol=0, atol=0)


def test_hyper_connection_state_lives_in_the_forward_not_in_the_module(tmp_path, fake_gdn):
    """At model level the HC state is per step; only the mixer is persistent."""
    root = tmp_path / "tiny-hc"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    model.attach_ple_tables(str(root))
    fill_expert_planes(model)
    lm = model.language_model

    layers = list(lm.model.layers)
    assert all(layer.attn_hyper_connection.hc_count == HC for layer in layers)
    assert all(layer.mlp_hyper_connection.hc_count == HC for layer in layers)
    # The final mixer collapses the streams and never produces an injection.
    mixer = lm.model.hyper_connection_mixer
    assert mixer is not None and mixer.use_combine is False

    tokens = 4
    ids = torch.randint(0, VOCAB, (tokens, ), generator=torch.Generator().manual_seed(12))
    out = model(input_ids=ids, positions=torch.arange(tokens))
    assert out.shape == (tokens, HIDDEN)

    # The pre-mixer multi-stream state is kept for the MTP draft (per step,
    # per token row) and is not a buffer/parameter of the module.
    multi = lm.model.get_mtp_hidden()
    assert multi is not None and multi.shape == (tokens, HC * HIDDEN)
    assert not [name for name in lm.model.state_dict() if "_last_mtp_hidden" in name]
    assert not [name for name, _ in lm.model.named_buffers() if "injection" in name
                or "block_output" in name]
    # hc_silu / hc_gate_mix / grouped norm are the pieces the math test above
    # pins; here we only check the dtype/rounding boundaries the model uses.
    assert hc_silu(torch.ones(2, HC, dtype=torch.bfloat16), HC).dtype == torch.bfloat16
    xn = hc_grouped_gemma_rmsnorm(torch.ones(2, HC * 8), torch.zeros(HC * 8), 1e-6, HC)
    assert xn.shape == (2, HC * 8)
    gate_mix = hc_gate_mix(torch.ones(2, HC * 8), torch.zeros(2, HC * 8), HC)
    assert gate_mix.shape == (2, 8)


# ---------------------------------------------------------------------------
# (9) PLE: explicit host-byte budget + measured per-token streaming
# ---------------------------------------------------------------------------
PLE_ROW_BYTES = PLE_HEAD_DIM // 2 + PLE_HEAD_DIM // 16
PLE_TABLE_BYTES = PLE_TABLE_ROWS * PLE_ROW_BYTES
PLE_NUM_HEADS = (TEXT_CONFIG["ngram_size"] - 1) * TEXT_CONFIG["heads_per_ngram"]


def test_ple_host_budget_is_explicit_and_enforced(tmp_path, monkeypatch):
    """Decision E's host-byte budget: charged at attach, enforced, reported."""
    monkeypatch.delenv("VLLM_PLE_HOST_BUDGET_GIB", raising=False)
    monkeypatch.delenv("VLLM_PLE_HOST_BUDGET_BYTES", raising=False)
    assert ple_host_budget_bytes() == qwen4_exp.PLE_DEFAULT_HOST_BUDGET_BYTES == 32 << 30
    monkeypatch.setenv("VLLM_PLE_HOST_BUDGET_GIB", "1.5")
    assert ple_host_budget_bytes() == int(1.5 * 2**30)
    monkeypatch.setenv("VLLM_PLE_HOST_BUDGET_BYTES", "1234")
    assert ple_host_budget_bytes() == 1234  # bytes wins over GiB
    monkeypatch.setenv("VLLM_PLE_HOST_BUDGET_BYTES", "0")
    with pytest.raises(ValueError):
        ple_host_budget_bytes()
    monkeypatch.delenv("VLLM_PLE_HOST_BUDGET_BYTES")
    monkeypatch.delenv("VLLM_PLE_HOST_BUDGET_GIB")

    root = tmp_path / "ple-budget"
    write_tiny_model(root, quant=False)
    offloader = PLEHostOffloader(budget_bytes=PLE_TABLE_BYTES)
    embedding = load_ple_host_embedding(root, layer_idx=1, heads_per_ngram=PLE_HEADS,
                                        ngram_size=TEXT_CONFIG["ngram_size"],
                                        offloader=offloader)
    assert isinstance(embedding, HostPLEEmbedding)
    assert offloader.used_bytes == PLE_TABLE_BYTES == embedding.host_bytes
    assert offloader.remaining_bytes == 0
    # Re-attaching the same layer with the same size is a no-op; a resize is
    # a programming error, and over-budget is refused before any mapping.
    offloader.reserve(1, PLE_TABLE_BYTES)
    with pytest.raises(RuntimeError, match="already reserved"):
        offloader.reserve(1, PLE_TABLE_BYTES + 1)
    too_small = PLEHostOffloader(budget_bytes=PLE_TABLE_BYTES - 1)
    with pytest.raises(PLEHostBudgetExceeded, match="host budget"):
        load_ple_host_embedding(root, layer_idx=1, heads_per_ngram=PLE_HEADS,
                                ngram_size=TEXT_CONFIG["ngram_size"], offloader=too_small)
    assert too_small.used_bytes == 0

    # Through the model: attach charges the same bytes and exposes the handle.
    model, _ = _build_model(root)
    assert model.language_model.attach_ple_tables(str(root)) == 1
    lm = model.language_model
    assert isinstance(lm.ple_offloader, PLEHostOffloader)
    assert lm.ple_host_bytes == PLE_TABLE_BYTES
    assert lm.ple_offloader.budget_bytes == 32 << 30
    assert lm.ple_offloader.lookups == 0
    # A model whose host budget cannot hold the table fails at attach time,
    # with no half-attached table.
    model2, _ = _build_model(root)
    model2.language_model._ple_offloader = PLEHostOffloader(budget_bytes=1)
    with pytest.raises(PLEHostBudgetExceeded):
        model2.attach_ple_tables(str(root))
    assert model2.language_model.ple_offloader.used_bytes == 0
    ple = list(model2.language_model.model.layers)[1].ple
    assert ple.ngram is None

    # The production load path (``load_weights`` -> ``_maybe_attach_ple_tables``)
    # attaches and charges the same budget without an explicit call.
    model3, _ = _build_model(root)
    assert model3.language_model.ple_offloader is None
    model3.language_model.load_weights([])
    assert model3.language_model.ple_offloader.used_bytes == PLE_TABLE_BYTES
    assert model3.language_model.ple_host_bytes == PLE_TABLE_BYTES


def test_ple_host_offload_switch_and_pinned_accounting(tmp_path, monkeypatch):
    """VLLM_PLE_CPU_OFFLOAD=0 keeps the legacy pinned path, still host-only."""
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "0")
    assert ple_cpu_offload_enabled() is False
    calls: list[dict] = []

    def fake_loader(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(host_mode="pinned", tables=[])

    import vllm_gaudi.models.qwen4exp.ple_runtime as ple_runtime

    monkeypatch.setattr(ple_runtime, "load_ple_ngram", fake_loader)
    offloader = PLEHostOffloader()
    result = load_ple_host_embedding(Path("/nonexistent"), layer_idx=2, heads_per_ngram=8,
                                     ngram_size=3, offloader=offloader)
    assert result.host_mode == "pinned"
    assert calls and calls[0]["pin_memory"] is True
    assert offloader.mode == "pinned"
    assert offloader.budget_bytes == 32 << 30


def test_ple_decode_step_transfers_only_token_rows(tmp_path, fake_gdn, monkeypatch):
    """A decode step moves ``tokens x 16 x head_dim`` values, never a table.

    The single transfer point is spied on, and the offloader's counters are
    asserted: one gather per PLE layer per step, exactly ``tokens x num_heads``
    rows read from the host tables, and only the decoded rows on their way to
    the model device. A meta-device lookup proves the table itself never
    follows: it stays host-resident while the result lands on the target
    device.
    """
    root = tmp_path / "ple-stream"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    assert model.language_model.attach_ple_tables(str(root)) == 1
    fill_expert_planes(model)
    lm = model.language_model
    offloader = lm.ple_offloader
    embedding = lm.model.layers[1].ple.ngram
    assert isinstance(embedding, HostPLEEmbedding)
    assert embedding.assert_host_resident() >= 2  # tables + hash metadata

    transfers: list[tuple[int, tuple[int, ...], str]] = []
    real_transfer = qwen4_exp._transfer_host_rows

    def spy(rows, device):
        transfers.append((int(rows.numel()), tuple(rows.shape),
                          "none" if device is None else str(device)))
        return real_transfer(rows, device)

    monkeypatch.setattr(qwen4_exp, "_transfer_host_rows", spy)

    # -- one decode token -------------------------------------------------
    consumed = model(input_ids=torch.tensor([5]), positions=torch.tensor([0]))
    assert consumed.shape == (1, HIDDEN)
    assert offloader.lookups == 1
    assert offloader.rows_gathered == PLE_NUM_HEADS == 16
    assert offloader.host_bytes_read == PLE_NUM_HEADS * PLE_ROW_BYTES
    assert offloader.device_bytes_sent == PLE_NUM_HEADS * PLE_HEAD_DIM * 2
    assert offloader.device_bytes_sent == 1 * 16 * PLE_HEAD_DIM * 2
    # The only device-bound transfer is the decoded row batch; the host-only
    # calls (device=None) are the per-shard gathers inside the lookup.
    def device_transfers() -> list[tuple[int, tuple[int, ...]]]:
        return [(numel, shape) for numel, shape, target in transfers if target != "none"]

    values_per_token = PLE_NUM_HEADS * PLE_HEAD_DIM  # 16 heads x head_dim
    assert device_transfers() == [(values_per_token, (1, PLE_HEADS * PLE_HEAD_DIM))]

    # -- three-token step: the traffic scales with the tokens, nothing else -
    before = (offloader.lookups, offloader.rows_gathered, offloader.device_bytes_sent)
    model(input_ids=torch.tensor([6, 7, 8]), positions=torch.tensor([0, 1, 2]))
    assert offloader.lookups == before[0] + 1
    assert offloader.rows_gathered == before[1] + 3 * PLE_NUM_HEADS
    assert offloader.device_bytes_sent == before[2] + 3 * PLE_NUM_HEADS * PLE_HEAD_DIM * 2
    assert device_transfers()[-1] == (3 * values_per_token, (3, PLE_HEADS * PLE_HEAD_DIM))
    # Never a whole table (64 rows x 16 values = 1024 values) in one transfer.
    assert max(numel for numel, _, _ in transfers) < PLE_TABLE_ROWS * PLE_HEAD_DIM

    # -- the table never follows the rows to the target device ------------
    ids = torch.zeros(2, PLE_NUM_HEADS, dtype=torch.int64)
    on_meta = embedding.lookup(ids, dtype=torch.bfloat16, device="meta")
    assert on_meta.device.type == "meta" and on_meta.shape == (2, PLE_HEADS * PLE_HEAD_DIM)
    assert device_transfers()[-1] == (2 * values_per_token, (2, PLE_HEADS * PLE_HEAD_DIM))
    assert [target for _, _, target in transfers][-1] == "meta"
    embedding.assert_host_resident()  # still a host object
    assert embedding.tables[0].packed.device.type == "cpu"
    assert embedding.tables[0].scales.device.type == "cpu"
    # Nothing that looks like the table is reachable from the module tree, so
    # a ``.to(device)`` on the model cannot move it.
    assert not [name for name, _ in model.named_buffers() if "ngram" in name]
    assert not [name for name in model.state_dict() if "ngram" in name]
    assert {p.device.type for p in model.parameters()} == {"cpu"}

    # -- the counters are readable in one line for the serve log -----------
    report = offloader.report()
    assert "host=" in report and "budget" in report and "rows=" in report


def test_h24_switches_and_gather_equivalence(tmp_path, monkeypatch):
    """H24: the fast host gather reads exactly the rows the mask gather did.

    The per-shard ``shard == s`` mask loop became one stable ``argsort`` plus
    contiguous per-shard segments. Both spellings must return byte-identical
    packed/scale rows (the decode is bit-exact, so a single differing byte
    would surface as a different embedding), and the switches must default to
    the adopted set — with the measured-and-rejected ``h2d`` item off, which
    the umbrella must not turn on.
    """
    from vllm_gaudi.models.qwen4exp.h24 import H24_ITEMS, UMBRELLA, h24_enabled

    for var in (UMBRELLA, *H24_ITEMS.values()):
        monkeypatch.delenv(var, raising=False)
    for item in ("gather", "norm", "spans", "conv"):
        assert h24_enabled(item) is True
    assert h24_enabled("h2d") is False  # lost at every regime, opt-in only
    monkeypatch.setenv(UMBRELLA, "0")
    for item in ("gather", "norm", "spans", "conv"):
        assert h24_enabled(item) is False
    monkeypatch.setenv(UMBRELLA, "1")
    assert h24_enabled("h2d") is False  # the umbrella never enables it
    monkeypatch.setenv(H24_ITEMS["h2d"], "1")
    assert h24_enabled("h2d") is True
    with pytest.raises(KeyError):
        h24_enabled("nope")

    root = tmp_path / "ple-gather"
    write_tiny_model(root, quant=False)
    # The tiny checkpoint has one shard, so this pins the equivalence on a
    # single segment; the real 128-shard multi-segment case (and the numpy
    # golden) is covered by tools/qwen4exp_flashnext/test_ple_runtime.py.
    embedding = load_ple_host_embedding(root, layer_idx=1, heads_per_ngram=8, ngram_size=3)
    # Repeated ids, ids in reverse order, and a shuffled mix: the sorted path
    # must handle ties and shard-order changes without reordering the output.
    ids = torch.tensor([[0, 1, 2, 3] * 4, [3, 2, 1, 0] * 4, [7, 7, 0, 5] * 4],
                       dtype=torch.int64)
    fast_packed, fast_scales = embedding._gather_sorted(ids)
    slow_packed, slow_scales = embedding._gather_masked(ids)
    assert torch.equal(fast_packed, slow_packed), "packed rows diverged"
    assert torch.equal(fast_scales, slow_scales), "scale rows diverged"
    # ...and the public lookup is the same tensor either way.
    monkeypatch.setenv(UMBRELLA, "0")
    masked = embedding.lookup(ids, dtype=torch.bfloat16, device="cpu")
    monkeypatch.setenv(UMBRELLA, "1")
    fast = embedding.lookup(ids, dtype=torch.bfloat16, device="cpu")
    assert torch.equal(masked, fast)


# ---------------------------------------------------------------------------
# (10) quantization wiring: the per-prefix choice is logged and unit-tested
# ---------------------------------------------------------------------------
def test_quant_dispatch_is_logged_per_prefix(tmp_path, caplog):
    """Decision D: each class logs once, and the log matches the built layers.

    experts -> NVFP4 packed residency, shared expert / QSA / GDN -> MXFP8 as
    Gaudi2 per-channel FP8, vision attention -> MXFP8, vision MLP -> W4A16
    dequantized to BF16, lm_head and hyper-connections -> unquantized.
    """
    from vllm_gaudi.ops import modelopt_csf

    root = tmp_path / "tiny-quant-log"
    write_tiny_model(root, quant=True)
    # Built unpatched: the log must report the class the decoder really built.
    with caplog.at_level("INFO", logger="vllm_gaudi.models.qwen4_exp"):
        model, _ = _build_model(root)
    lm = model.language_model

    kinds = lm.report_quant_dispatch()  # also re-logs, so the helper is exercised twice
    assert kinds == {
        "experts": modelopt_csf.KIND_NVFP4_PACKED,
        "shared_expert": modelopt_csf.KIND_MXFP8,
        "gdn": modelopt_csf.KIND_MXFP8,
        "qsa": modelopt_csf.KIND_MXFP8,
    }
    messages = [record.getMessage() for record in caplog.records]
    dispatch_lines = [message for message in messages if "MIXED_PRECISION dispatch" in message]
    assert dispatch_lines, messages
    assert any("(language)" in line for line in dispatch_lines)
    assert any(modelopt_csf.KIND_NVFP4_PACKED in line for line in dispatch_lines)
    assert any(modelopt_csf.KIND_MXFP8 in line for line in dispatch_lines)
    # The linear half's identity is logged with the same construction.
    assert any("GDN linear attention -> vllm_gaudi.models.qwen3_5.HPUGatedDeltaNetAttention"
               in message for message in messages)

    # The vision tower is not built for --language-model-only, but the same
    # report states the classes its declared prefixes resolve to.
    extras = visual_dispatch_samples(lm.quant_config)
    assert extras == {
        "visual_attn_qkv": "model.visual.blocks.0.attn.qkv",
        "visual_mlp_fc1": "model.visual.blocks.0.mlp.linear_fc1",
    }
    with caplog.at_level("INFO", logger="vllm_gaudi.models.qwen4_exp"):
        vision = quant_dispatch_report(
            lm.quant_config,
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
    assert any("(vision)" in record.getMessage() for record in caplog.records)


def test_qsa_lazy_pool_matches_eager_pool(tmp_path, fake_gdn, monkeypatch):
    """H30: below the token budget nothing is pooled; past it the rows equal per-commit pooling."""
    root = tmp_path / "tiny-qsa-lazy"
    write_tiny_model(root, quant=False)
    model, _ = _build_model(root)
    qsa = model.language_model.model.layers[3].self_attn
    budget = qsa.indexer.token_topk
    torch.manual_seed(0)
    prefill = torch.randn(budget - 1, HIDDEN, dtype=torch.bfloat16)
    steps = [torch.randn(1, HIDDEN, dtype=torch.bfloat16) for _ in range(6)]

    def run(lazy: str):
        monkeypatch.setenv("VLLM_QWEN4EXP_QSA_LAZY_POOL", lazy)
        qsa.reset_bringup_cache()
        outs = [qsa(hidden_states=prefill, positions=torch.arange(budget - 1),
                    query_start_loc=torch.tensor([0, budget - 1]))]
        pooled_below = qsa._bringup[0].pooled.shape[0]
        for i, step in enumerate(steps):
            pos = budget - 1 + i
            outs.append(qsa(hidden_states=step, positions=torch.tensor([pos]),
                            query_start_loc=torch.tensor([0, 1])))
        return outs, pooled_below, qsa._bringup[0].pooled.clone()

    eager_outs, eager_below, eager_pooled = run("0")
    lazy_outs, lazy_below, lazy_pooled = run("1")
    assert eager_below == (budget - 1) // qsa.indexer.compress_ratio
    assert lazy_below == 0
    assert torch.equal(lazy_pooled, eager_pooled)
    for lazy, eager in zip(lazy_outs, eager_outs):
        assert torch.equal(lazy, eager)



def test_stable_topk_desc_matches_stable_argsort():
    """H31: the prefill block selection's top-k equals the first k of a stable descending argsort."""
    from vllm_gaudi.models.qwen4exp.qsa_math import stable_topk_desc

    gen = torch.Generator().manual_seed(0)
    for trial in range(24):
        rows, cols, k = 9, int(torch.randint(600, 3000, (1,), generator=gen)), 512
        scores = torch.randn(rows, cols, generator=gen)
        # relu zeros and quarter steps give the exact ties the real indexer produces
        scores = torch.relu(scores) if trial % 2 else torch.round(scores * 4) / 4
        if trial % 3 == 0:
            cut = torch.randint(0, cols, (rows, 1), generator=gen)
            scores = scores.masked_fill(torch.arange(cols)[None] >= cut, float("-inf"))
        expected = torch.argsort(scores, dim=-1, stable=True, descending=True)[:, :k]
        assert torch.equal(stable_topk_desc(scores, k), expected)


def test_views_of_tracks_storage_not_address():
    """H31: a merged projection stays valid while its parameters are its row blocks, wherever it lives."""
    from vllm_gaudi.models.qwen4exp.merged_weights import views_of

    merged = torch.randn(7, 4)
    first, second = merged[:3], merged[3:]
    assert views_of(merged, first, second)
    assert not views_of(merged, second, first)  # wrong order
    assert not views_of(merged, first)  # does not cover the concatenation
    assert not views_of(merged, first.clone(), second)  # a separate copy is not a view
    assert not views_of(merged, merged[:3, :2], merged[3:, :2])  # column slices
    assert not views_of(None, first, second)
    # A different concatenation of equal values is not the cached one.
    assert not views_of(merged.clone(), first, second)
