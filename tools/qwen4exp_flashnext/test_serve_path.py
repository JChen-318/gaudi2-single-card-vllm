# SPDX-License-Identifier: Apache-2.0
"""Serve-path tests: standard config parsing and the converted-checkpoint loader.

1. The real converted checkpoint parses through ``get_config`` /
   ``ModelConfig`` with no dummy-config swap: the plugin-registered
   ``PretrainedConfig`` facade hands vLLM the text shim, vendor layer
   accounting sees the raw ``full_attention`` dialect (12 attention + 36
   linear layers), MoE metadata resolves, and the FP4 quantization_config is
   stripped for the BF16 bring-up.
2. ``--load-format qwen4exp_plain`` streams a miniature converted checkpoint
   (same layout: ``tensors/`` + index) into a miniature model: direct
   tensors, the q/k/v, shared gate/up, and merged HC down+inject stacks, the
   per-expert FusedMoE shards, and the vision-name filter for text-only
   targets. The same loader feeds a miniature ``Qwen4ExpMTP`` draft from the
   ``mtp.*`` group only, leaving the shared embed/LM head untouched.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")
# Per-parameter HPU sync after each weight load needs a driver; this host
# has none. Production keeps the default (true).
os.environ.setdefault("VLLM_WEIGHT_LOAD_FORCE_SYNC", "false")

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

CKPT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()


def _register() -> None:
    import vllm_gaudi

    vllm_gaudi.register_models()


def test_real_config_path() -> None:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import get_config, get_hf_text_config

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.hf_config import Qwen4ExpHfConfig

    cfg = get_config(str(CKPT), trust_remote_code=False)
    assert isinstance(cfg, Qwen4ExpHfConfig), type(cfg)
    assert cfg.model_type == "qwen4_exp"
    assert cfg.quantization_config is None  # stripped for the BF16 bring-up
    text = get_hf_text_config(cfg)
    assert isinstance(text, Qwen4ExpTextConfig), type(text)
    assert text.num_attention_heads == 24
    assert getattr(text, "eos_token_id") == 248044
    counts = {t: text.layer_types.count(t) for t in set(text.layer_types)}
    assert counts == {"full_attention": 12, "linear_attention": 36}, counts

    model_config = ModelConfig(
        model=str(CKPT), tokenizer=str(CKPT), dtype="bfloat16", seed=0, max_model_len=512
    )
    # ModelConfig re-parses the config internally, so compare fields.
    mc_text = model_config.hf_text_config
    assert isinstance(mc_text, Qwen4ExpTextConfig), type(mc_text)
    assert mc_text.layer_types == text.layer_types
    assert mc_text.num_attention_heads == 24
    assert getattr(mc_text, "eos_token_id") == 248044
    assert model_config.hf_config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert model_config.is_moe
    assert model_config.get_num_experts() == 512
    vllm_config = VllmConfig(model_config=model_config)
    pc = vllm_config.parallel_config
    n_attn = model_config.get_num_layers_by_block_type(pc, "attention")
    n_lin = model_config.get_num_layers_by_block_type(pc, "linear_attention")
    assert (n_attn, n_lin) == (12, 36), (n_attn, n_lin)
    cls = ModelRegistry._try_load_model_cls("Qwen4ExpForConditionalGeneration")
    assert cls is not None and cls.__name__ == "Qwen4ExpForConditionalGeneration"
    print(f"  real config: parsed, 12+36 layers, 512 experts, arch resolved: OK")


def _tiny_text_config() -> dict:
    return {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
        "full_attention_interval": 1,
        "layer_types": ["full_attention", "full_attention"],
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": True,
        "hc_count": 4,
        "hc_lowrank": 8,
        "ple_layer_ids": [],
        "mtp_num_hidden_layers": 1,
        "eos_token_id": 7,
        "hidden_act": "silu",
        "attn_output_gate": True,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        },
    }


_DIST_INITIALIZED = False


def _init_dist(vllm_config):
    global _DIST_INITIALIZED
    from vllm.config import set_current_vllm_config

    ctx = set_current_vllm_config(vllm_config)
    ctx.__enter__()  # caller must keep the reference alive for the whole test
    if not _DIST_INITIALIZED:
        from vllm.distributed import init_distributed_environment, initialize_model_parallel

        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        import vllm.platforms as vp

        vp.current_platform.current_device = lambda: torch.device("cpu")
        _DIST_INITIALIZED = True
    return ctx


def _checkpoint_view(cfg, state, ckpt_name, expert_prefix, num_layers=None) -> dict[str, torch.Tensor]:
    """Checkpoint-side tensors for a filled model ``state``.

    Splits the merged qkv / shared gate_up / HC down+inject stacks the way
    the checkpoint ships them and synthesizes per-expert FusedMoE shards.
    ``ckpt_name`` may return None to drop a tensor (e.g. the draft's shared
    embed/LM head, which never ship in the mtp.* group).
    """
    q_rows = cfg.num_attention_heads * cfg.head_dim * 2  # attn_output_gate
    kv_rows = cfg.num_key_value_heads * cfg.head_dim
    inter = cfg.shared_expert_intermediate_size
    low, hc = cfg.hc_lowrank, cfg.hc_count
    tensors: dict[str, torch.Tensor] = {}
    skip_suffix = ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")
    for name, value in state.items():
        if name.endswith(skip_suffix) or ".mlp.experts.routed_experts." in name:
            continue
        mapped = ckpt_name(name)
        if mapped is None:
            continue
        if name.endswith("self_attn.qkv_proj.weight"):
            base = name[: -len("qkv_proj.weight")]
            prefix = ckpt_name(base)
            tensors[prefix + "q_proj.weight"] = value[:q_rows].contiguous()
            tensors[prefix + "k_proj.weight"] = value[q_rows:q_rows + kv_rows].contiguous()
            tensors[prefix + "v_proj.weight"] = value[q_rows + kv_rows:].contiguous()
        elif name.endswith("mlp.shared_expert.gate_up_proj.weight"):
            base = name[: -len("gate_up_proj.weight")]
            prefix = ckpt_name(base)
            tensors[prefix + "gate_proj.weight"] = value[:inter].contiguous()
            tensors[prefix + "up_proj.weight"] = value[inter:].contiguous()
        elif name.endswith("input_mix_weight_down_block_inject.weight"):
            base = name[: -len("input_mix_weight_down_block_inject.weight")]
            prefix = ckpt_name(base)
            tensors[prefix + "input_mix_weight_down.weight"] = value[:low].contiguous()
            tensors[prefix + "block_inject_weight.weight"] = value[low:low + hc].contiguous()
        else:
            # Embedding tables are padded in the model (divisible-by default)
            # but ship unpadded in the checkpoint, like the real one.
            if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
                value = value[:cfg.vocab_size]
            tensors[mapped] = value.contiguous()

    for layer in range(cfg.num_hidden_layers if num_layers is None else num_layers):
        for expert in range(cfg.num_experts):
            pre = f"{expert_prefix}.layers.{layer}.mlp.experts.{expert}."
            hidden = cfg.hidden_size
            inter = cfg.shared_expert_intermediate_size
            # The converted checkpoint ships experts as packed NVFP4 groups
            # (U8 nibbles, low = even k / high = odd k; F8_E4M3 per-16 plane;
            # F32 weight_scale_2), never dense BF16 — the packed store refuses
            # scale-less experts (they would stay _pending forever).
            gen = torch.Generator().manual_seed(1000 * (layer + 1) + expert)
            for kind, rows, cols in (
                ("gate_proj", inter, hidden),
                ("up_proj", inter, hidden),
                ("down_proj", hidden, inter),
            ):
                codes = torch.randint(0, 16, (rows, cols), generator=gen, dtype=torch.int32)
                tensors[pre + f"{kind}.weight"] = (
                    codes[:, 0::2] | (codes[:, 1::2] << 4)
                ).to(torch.uint8)
                # 0x7F / 0xFF are e4m3fn NaN. A NaN scale makes the dequant
                # round-trip fail torch.equal even when every finite value matches.
                mag = torch.randint(0, 127, (rows, cols // 16), generator=gen, dtype=torch.uint8)
                sign = torch.randint(0, 2, (rows, cols // 16), generator=gen, dtype=torch.uint8)
                tensors[pre + f"{kind}.weight_scale"] = (mag | (sign << 7)).view(torch.float8_e4m3fn)
                tensors[pre + f"{kind}.weight_scale_2"] = torch.tensor(
                    [0.5 + 0.25 * expert], dtype=torch.float32)
    return tensors


def test_plain_loader() -> None:
    from safetensors.torch import load_file
    from vllm.config import LoadConfig, ModelConfig, VllmConfig
    from vllm.model_executor.model_loader import get_model_loader

    from vllm_gaudi.models.qwen4exp.loader import Qwen4ExpPlainModelLoader
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpForCausalLM

    root = Path(tempfile.mkdtemp()) / "mini-plain"
    (root / "tensors").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForCausalLM"],
        "torch_dtype": "bfloat16",
        "text_config": _tiny_text_config(),
    }))

    model_config = ModelConfig(
        model=str(root), tokenizer=str(root), dtype="bfloat16", seed=0, max_model_len=64
    )
    vllm_config = VllmConfig(model_config=model_config)
    torch.set_default_dtype(torch.bfloat16)
    keep_ctx = _init_dist(vllm_config)  # noqa: F841 - holds the config context
    model = Qwen4ExpForCausalLM(vllm_config=vllm_config, prefix="")
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    src = {k: v.clone() for k, v in model.state_dict().items()}

    # -- build the checkpoint-side view of the same weights ----------------
    cfg = model_config.hf_text_config
    q_rows = cfg.num_attention_heads * cfg.head_dim * 2  # attn_output_gate
    inter = cfg.shared_expert_intermediate_size
    low, hc = cfg.hc_lowrank, cfg.hc_count

    def ckpt_name(model_name: str) -> str:
        # state_dict lives in the ForCausalLM namespace ("model.…"); the
        # checkpoint nests the same tree under "model.language_model.".
        if model_name.startswith("model."):
            return "model.language_model." + model_name[len("model."):]
        return model_name  # lm_head.weight ships unprefixed

    tensors = _checkpoint_view(cfg, src, ckpt_name, "model.language_model")
    # Text-only target: tower names must be filtered by the loader, not choke it.
    tensors["model.visual.block.weight"] = torch.randn(4, 4)

    shard = root / "tensors" / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {name: shard.name for name in tensors},
    }))
    on_disk = {k: v.clone() for k, v in load_file(str(shard)).items()}

    loader = get_model_loader(LoadConfig(load_format="qwen4exp_plain"))
    assert isinstance(loader, Qwen4ExpPlainModelLoader), type(loader)
    loader.download_model(model_config)
    loader.load_weights(model, model_config)

    with torch.no_grad():
        vocab = cfg.vocab_size
        assert torch.equal(model.lm_head.weight[:vocab], on_disk["lm_head.weight"])
        assert torch.equal(model.model.embed_tokens.weight[:vocab], on_disk[
            "model.language_model.embed_tokens.weight"])
        attn = model.model.layers[1].self_attn
        assert torch.equal(attn.qkv_proj.weight[:q_rows], on_disk[
            "model.language_model.layers.1.self_attn.q_proj.weight"])
        shared = model.model.layers[1].mlp.shared_expert
        assert torch.equal(shared.gate_up_proj.weight[:inter], on_disk[
            "model.language_model.layers.1.mlp.shared_expert.gate_proj.weight"])
        hc_mod = model.model.layers[1].attn_hyper_connection
        merged = hc_mod.input_mix_weight_down_block_inject.weight
        assert torch.equal(merged[:low], on_disk[
            "model.language_model.layers.1.attn_hyper_connection.input_mix_weight_down.weight"])
        assert torch.equal(merged[low:low + hc], on_disk[
            "model.language_model.layers.1.attn_hyper_connection.block_inject_weight.weight"])
        runner = model.model.layers[1].mlp.experts
        from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts
        from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

        assert isinstance(runner, PackedRoutedExperts), type(runner)
        for expert in range(cfg.num_experts):
            pre = f"model.language_model.layers.1.mlp.experts.{expert}."
            for kind in ("gate_proj", "up_proj", "down_proj"):
                assert torch.equal(
                    getattr(runner, f"{kind}_packed")[expert],
                    on_disk[pre + f"{kind}.weight"])
                assert torch.equal(
                    getattr(runner, f"{kind}_scale")[expert],
                    on_disk[pre + f"{kind}.weight_scale"].view(torch.uint8))
                assert float(getattr(runner, f"{kind}_scale2")[expert]) == float(
                    on_disk[pre + f"{kind}.weight_scale_2"])
            # Host dequant round-trip: scale2 folded after the BF16 cast,
            # exactly like dequant_indexed's CPU path.
            got = runner.dequant("gate_proj", torch.tensor([expert]))[0]
            want = dequant_nvfp4(
                on_disk[pre + "gate_proj.weight"],
                on_disk[pre + "gate_proj.weight_scale"], 1.0)
            want = want * torch.tensor(
                float(on_disk[pre + "gate_proj.weight_scale_2"]), dtype=want.dtype)
            assert torch.equal(got, want)
        assert not hasattr(model, "visual")
    print(f"  qwen4exp_plain: {len(tensors)} tensors loaded, stacks + experts verified: OK")


def test_plain_loader_draft() -> None:
    from safetensors.torch import load_file
    from vllm.config import LoadConfig, ModelConfig, VllmConfig

    from vllm_gaudi.models.qwen4exp.loader import Qwen4ExpPlainModelLoader
    from vllm_gaudi.models.qwen4exp.mtp import Qwen4ExpMTP

    root = Path(tempfile.mkdtemp()) / "mini-plain-draft"
    (root / "tensors").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpMTP"],
        "torch_dtype": "bfloat16",
        "text_config": _tiny_text_config(),
    }))
    model_config = ModelConfig(
        model=str(root), tokenizer=str(root), dtype="bfloat16", seed=0, max_model_len=64
    )
    vllm_config = VllmConfig(model_config=model_config)
    keep_ctx = _init_dist(vllm_config)  # noqa: F841 - holds the config context
    draft = Qwen4ExpMTP(vllm_config=vllm_config, prefix="")
    with torch.no_grad():
        for p in draft.parameters():
            p.normal_(0, 0.02)
    src = {k: v.clone() for k, v in draft.state_dict().items()}
    cfg = model_config.hf_text_config

    def ckpt_name(model_name: str) -> str | None:
        # Draft weights ship under "mtp.*"; the embed and LM head are shared
        # from the target by the proposer and never appear on disk.
        if model_name == "lm_head.weight" or model_name.endswith("embed_tokens.weight"):
            return None
        if model_name.startswith("model."):
            return "mtp." + model_name[len("model."):]
        return model_name

    tensors = _checkpoint_view(cfg, src, ckpt_name, "mtp", num_layers=1)
    shard = root / "tensors" / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {name: shard.name for name in tensors},
    }))
    on_disk = {k: v.clone() for k, v in load_file(str(shard)).items()}

    lm_before = draft.lm_head.weight.clone()
    embed_before = draft.model.embed_tokens.weight.clone()
    loader = Qwen4ExpPlainModelLoader(LoadConfig(load_format="qwen4exp_plain"))
    loader.download_model(model_config)
    loader.load_weights(draft, model_config)

    with torch.no_grad():
        assert torch.equal(draft.model.fc_embedding.weight, on_disk["mtp.fc_embedding.weight"])
        assert torch.equal(draft.model.fc_hidden.weight, on_disk["mtp.fc_hidden.weight"])
        assert torch.equal(
            draft.model.hyper_connection_mixer.input_mix_weight_down.weight,
            on_disk["mtp.hyper_connection_mixer.input_mix_weight_down.weight"],
        )
        q_rows = cfg.num_attention_heads * cfg.head_dim * 2
        attn = draft.model.layers[0].self_attn
        assert torch.equal(attn.qkv_proj.weight[:q_rows], on_disk[
            "mtp.layers.0.self_attn.q_proj.weight"])
        low, hc = cfg.hc_lowrank, cfg.hc_count
        merged = draft.model.layers[0].attn_hyper_connection.input_mix_weight_down_block_inject.weight
        assert torch.equal(merged[:low], on_disk[
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight"])
        assert torch.equal(merged[low:low + hc], on_disk[
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight"])
        runner = draft.model.layers[0].mlp.experts
        from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts
        from vllm_gaudi.models.qwen4exp.weights import dequant_nvfp4

        assert isinstance(runner, PackedRoutedExperts), type(runner)
        for expert in range(cfg.num_experts):
            pre = f"mtp.layers.0.mlp.experts.{expert}."
            for kind in ("gate_proj", "up_proj", "down_proj"):
                assert torch.equal(
                    getattr(runner, f"{kind}_packed")[expert],
                    on_disk[pre + f"{kind}.weight"])
                assert torch.equal(
                    getattr(runner, f"{kind}_scale")[expert],
                    on_disk[pre + f"{kind}.weight_scale"].view(torch.uint8))
                assert float(getattr(runner, f"{kind}_scale2")[expert]) == float(
                    on_disk[pre + f"{kind}.weight_scale_2"])
            got = runner.dequant("down_proj", torch.tensor([expert]))[0]
            want = dequant_nvfp4(
                on_disk[pre + "down_proj.weight"],
                on_disk[pre + "down_proj.weight_scale"], 1.0)
            want = want * torch.tensor(
                float(on_disk[pre + "down_proj.weight_scale_2"]), dtype=want.dtype)
            assert torch.equal(got, want)
        # The proposer shares these from the target; the draft load must not
        # touch them.
        assert torch.equal(draft.lm_head.weight, lm_before)
        assert torch.equal(draft.model.embed_tokens.weight, embed_before)
    print(f"  qwen4exp_plain draft: {len(tensors)} mtp.* tensors loaded: OK")


def test_plain_loader_for_cond_gen() -> None:
    from safetensors.torch import load_file
    from vllm.config import LoadConfig, ModelConfig, VllmConfig

    from vllm_gaudi.models.qwen4exp.loader import Qwen4ExpPlainModelLoader
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpForConditionalGeneration

    root = Path(tempfile.mkdtemp()) / "mini-plain-cond"
    (root / "tensors").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "text_config": _tiny_text_config(),
        # Never built with language_model_only, but the shim still parses it.
        "vision_config": {
            "depth": 1, "hidden_size": 8, "num_heads": 2,
            "intermediate_size": 8, "patch_size": 2, "spatial_merge_size": 1,
            "temporal_patch_size": 1, "out_hidden_size": 32,
            "num_position_embeddings": 4,
        },
    }))
    model_config = ModelConfig(
        model=str(root), tokenizer=str(root), dtype="bfloat16", seed=0, max_model_len=64
    )
    model_config.multimodal_config.language_model_only = True
    vllm_config = VllmConfig(model_config=model_config)
    keep_ctx = _init_dist(vllm_config)  # noqa: F841 - holds the config context
    model = Qwen4ExpForConditionalGeneration(vllm_config=vllm_config, prefix="")
    assert model.language_model_only
    assert not hasattr(model, "visual"), "text-only target must not build the tower"
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.02)
    src = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = model_config.hf_text_config

    def ckpt_name(model_name: str) -> str | None:
        # state_dict namespace is "language_model.model.*"/"language_model.lm_head";
        # the checkpoint nests the trunk under "model.language_model." and ships
        # lm_head at the top level.
        if model_name == "language_model.lm_head.weight":
            return "lm_head.weight"
        if model_name.startswith("language_model.model."):
            return "model.language_model." + model_name[len("language_model.model."):]
        return None  # nothing else maps (no tower on a text-only target)

    tensors = _checkpoint_view(cfg, src, ckpt_name, "model.language_model")
    # Tower weights stream through the loader; the mapper must drop them.
    tensors["model.visual.blocks.0.mlp.fc1.weight"] = torch.randn(4, 4)
    tensors["model.visual.merger.mlp.0.weight"] = torch.randn(4, 4)
    shard = root / "tensors" / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {name: shard.name for name in tensors},
    }))
    on_disk = {k: v.clone() for k, v in load_file(str(shard)).items()}

    loader = Qwen4ExpPlainModelLoader(LoadConfig(load_format="qwen4exp_plain"))
    loader.download_model(model_config)
    loader.load_weights(model, model_config)

    with torch.no_grad():
        vocab = cfg.vocab_size
        lm = model.language_model
        assert torch.equal(lm.lm_head.weight[:vocab], on_disk["lm_head.weight"])
        assert torch.equal(lm.model.embed_tokens.weight[:vocab], on_disk[
            "model.language_model.embed_tokens.weight"])
        q_rows = cfg.num_attention_heads * cfg.head_dim * 2
        attn = lm.model.layers[1].self_attn
        assert torch.equal(attn.qkv_proj.weight[:q_rows], on_disk[
            "model.language_model.layers.1.self_attn.q_proj.weight"])
        runner = lm.model.layers[1].mlp.experts
        from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts

        assert isinstance(runner, PackedRoutedExperts), type(runner)
        pre = "model.language_model.layers.1.mlp.experts.0."
        for kind in ("gate_proj", "up_proj", "down_proj"):
            assert torch.equal(
                getattr(runner, f"{kind}_packed")[0],
                on_disk[pre + f"{kind}.weight"])
            assert torch.equal(
                getattr(runner, f"{kind}_scale")[0],
                on_disk[pre + f"{kind}.weight_scale"].view(torch.uint8))
            assert float(getattr(runner, f"{kind}_scale2")[0]) == float(
                on_disk[pre + f"{kind}.weight_scale_2"])
        assert not hasattr(model, "visual")
    print(f"  qwen4exp_plain for-cond-gen (language_model_only): "
          f"{len(tensors)} tensors, tower dropped, lm_head moved: OK")


def test_real_layout_discovery() -> None:
    from vllm.config import LoadConfig, ModelConfig

    from vllm_gaudi.models.qwen4exp.loader import Qwen4ExpPlainModelLoader

    model_config = ModelConfig(
        model=str(CKPT), tokenizer=str(CKPT), dtype="bfloat16", seed=0, max_model_len=512
    )
    loader = Qwen4ExpPlainModelLoader(LoadConfig(load_format="qwen4exp_plain"))
    loader.download_model(model_config)  # index + 21 shards, no weight IO
    shards = loader._shards(model_config)
    assert len(shards) == 21, len(shards)
    print(f"  real layout: index + {len(shards)} shards discovered: OK")


def main() -> int:
    _register()
    test_real_config_path()
    test_plain_loader()
    test_plain_loader_draft()
    test_plain_loader_for_cond_gen()
    test_real_layout_discovery()
    print("test_serve_path: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
