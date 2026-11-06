# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Qwen4Exp MTP draft head.

1. Fusion adds the projected token embedding to every HC stream.
2. A tiny draft forward returns ``(sample [T, H], multi [T, hc*H])`` and a
   second step accepts that multi stream.
3. Every non-expert ``mtp.*`` checkpoint name maps onto a draft parameter.
   Expert shards are counted (512 x 3). Embed and LM head are not in the
   MTP group; ``share_embeddings`` aliases the target modules.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

CKPT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()


def _tiny_config():
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        vocab_size=32,
        rms_norm_eps=1e-6,
        layer_types=["indexed_attention"],
        indexer_budget=4,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[],
        mtp_num_hidden_layers=1,
        tie_word_embeddings=False,
        extras={
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "rope_type": "default",
            },
            "hidden_act": "silu",
            "attn_output_gate": True,
        },
    )


def _vllm_config(cfg):
    from vllm.config import ModelConfig, VllmConfig

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
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype="float32", seed=0, max_model_len=64
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    return VllmConfig(model_config=model_config)


def _init_dist():
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


def test_fusion_adds_embedding_to_every_stream(pred) -> None:
    cfg = pred.config
    # Identity projections, zero Gemma weights (norm still divides by rms).
    with torch.no_grad():
        pred.fc_embedding.weight.copy_(torch.eye(cfg.hidden_size))
        pred.fc_hidden.weight.copy_(torch.eye(cfg.hidden_size))
        pred.pre_fc_norm_embedding.weight.zero_()
        pred.pre_fc_norm_hidden.weight.zero_()
    dtype = pred.fc_embedding.weight.dtype
    embeds = torch.randn(3, cfg.hidden_size, dtype=dtype)
    hidden = torch.randn(3, cfg.hc_count * cfg.hidden_size, dtype=dtype)
    fused = pred.fuse(embeds, hidden)

    embed_n = pred.pre_fc_norm_embedding(embeds)
    embed_p = embed_n @ pred.fc_embedding.weight.T
    streams = hidden.view(3, cfg.hc_count, cfg.hidden_size)
    hidden_n = pred.pre_fc_norm_hidden(streams.flatten(-2)).view_as(streams)
    hidden_p = hidden_n @ pred.fc_hidden.weight.T
    manual = (embed_p.unsqueeze(-2) + hidden_p).flatten(-2)
    assert fused.shape == (3, cfg.hc_count * cfg.hidden_size)
    assert torch.allclose(fused, manual, rtol=1e-5, atol=1e-5)
    # The embedding residual is present on every stream, not just stream 0.
    # BF16 subtraction is not exact, so compare the reconstructed sum.
    per_stream = fused.view(3, cfg.hc_count, cfg.hidden_size)
    rebuilt = hidden_p + embed_p.unsqueeze(-2)
    assert torch.allclose(per_stream, rebuilt, rtol=1e-2, atol=1e-2)
    print("  fusion adds embedding to every HC stream: OK")


def test_draft_forward_and_share(draft) -> None:
    cfg = draft.config
    assert draft.model.layers[0].ple is None
    assert draft.model.layers[0].layer_type == "indexed_attention"
    target_embed = draft.model.embed_tokens
    target_head = draft.lm_head
    draft.share_embeddings(target_embed, target_head)
    assert draft.model.embed_tokens is target_embed
    assert draft.lm_head is target_head

    dtype = draft.model.fc_embedding.weight.dtype
    with torch.no_grad():
        # Unconditional: torch.empty remnants can be finite-but-huge and
        # blow the BF16 forward up even though isfinite() passes.
        for param in draft.parameters():
            param.normal_(0, 0.02)
    # FusedMoE's custom op is HPU-only. The draft wiring under test is fusion,
    # QSA, and the HC mixer; the expert compute is unchanged from the trunk.
    draft.model.layers[0].mlp.forward = lambda hidden_states, already_sequence_parallel=False: (
        hidden_states.new_zeros(hidden_states.shape)
    )
    tokens = torch.randint(0, cfg.vocab_size, (4,))
    positions = torch.arange(4)
    hidden = torch.randn(4, cfg.hc_count * cfg.hidden_size, dtype=dtype)
    sample, multi = draft(tokens, positions, hidden_states=hidden)
    assert sample.shape == (4, cfg.hidden_size), sample.shape
    assert multi.shape == (4, cfg.hc_count * cfg.hidden_size), multi.shape
    assert torch.isfinite(sample).all() and torch.isfinite(multi).all()

    positions2 = torch.tensor([4])
    sample2, multi2 = draft(
        tokens[:1], positions2, hidden_states=multi[:1], spec_step_idx=1
    )
    assert sample2.shape == (1, cfg.hidden_size)
    assert multi2.shape == (1, cfg.hc_count * cfg.hidden_size)
    assert draft.model.layers[0].self_attn._bringup[0].raw_keys.shape[0] == 5
    print(f"  draft forward {tuple(sample.shape)} / {tuple(multi.shape)} + step2 carry: OK")


def test_checkpoint_names(draft) -> None:
    from vllm_gaudi.models.qwen4exp.mtp import remap_mtp_weight_name
    params = set(draft.state_dict())
    mapper = draft.hf_to_vllm_mapper
    weight_map = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    names = [
        n for n in weight_map
        if n.startswith("mtp.")
        and not n.endswith((".weight_scale", ".weight_scale_2", ".input_scale"))
    ]
    missing = []
    experts = 0
    for name in names:
        if ".mlp.experts." in name and name.endswith(
            (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
        ):
            experts += 1
            continue
        remapped = remap_mtp_weight_name(name)
        assert remapped is not None, name
        mapped = mapper._map_name(remapped)
        if mapped is None or mapped not in params:
            missing.append((name, remapped, mapped))
    assert not missing, missing[:8]
    assert experts == 512 * 3, experts
    # Embed / head are intentionally absent from the MTP group.
    assert not any(n.startswith("mtp.embed_tokens") or "shared_head" in n for n in names)
    print(f"  checkpoint names: {len(names) - experts} params + {experts} expert shards: OK")


def test_target_stashes_multi_stream() -> None:
    from vllm.config import get_current_vllm_config

    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpModel

    cfg = _tiny_config()
    model = Qwen4ExpModel(vllm_config=get_current_vllm_config(), prefix="model")
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.02)
    model.layers[0].mlp.forward = lambda hidden_states, already_sequence_parallel=False: (
        hidden_states.new_zeros(hidden_states.shape)
    )
    tokens = torch.randint(0, cfg.vocab_size, (3,))
    positions = torch.arange(3)
    hidden = model(tokens, positions)
    multi = model.get_mtp_hidden()
    assert hidden.shape == (3, cfg.hidden_size), hidden.shape
    assert multi is not None and multi.shape == (3, cfg.hc_count * cfg.hidden_size), (
        None if multi is None else multi.shape
    )
    assert torch.isfinite(hidden).all() and torch.isfinite(multi).all(), (
        hidden.float().abs().max() if hidden is not None else None,
        None if multi is None else multi.float().abs().max(),
    )
    print(f"  target stashes multi stream {tuple(multi.shape)}: OK")


def main() -> int:
    from vllm.config import set_current_vllm_config

    from vllm_gaudi.models.qwen4exp.mtp import Qwen4ExpMTP

    cfg = _tiny_config()
    vllm_config = _vllm_config(cfg)
    # vLLM serve sets the default dtype from the model config before init.
    # HC weights are explicitly bf16; leaving the default at fp32 splits the graph.
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(vllm_config):
        _init_dist()
        draft = Qwen4ExpMTP(vllm_config=vllm_config, prefix="")
        test_fusion_adds_embedding_to_every_stream(draft.model)
        test_draft_forward_and_share(draft)
        test_checkpoint_names(draft)
        test_target_stashes_multi_stream()
    print("test_mtp: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
