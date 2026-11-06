# SPDX-License-Identifier: Apache-2.0
"""CPU construction test for the Qwen4Exp HPU port model skeleton.

Builds ``Qwen4ExpForCausalLM`` from the real checkpoint config (via the local
shim) under a gloo/distributed test harness and validates the decoder layer
graph: 36 GDN + 12 (QSA-type) full-attention layers, PLE on layer 1
(ple_layer_ids=[2] -> layer_idx+1), MoE expert counts, hyper-connection
geometry, and the mamba state-shape protocol.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

# No HPU driver on this host: force the vllm-gaudi extension to select the
# CPU code paths for this construction test.
os.environ.setdefault("VLLM_HW", "cpu")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

from vllm_gaudi.models.qwen4exp.config import Qwen4ExpConfig  # noqa: E402

SNAP = Path.home() / (
    ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/"
    "snapshots"
)


def find_config() -> Path:
    if SNAP.is_dir():
        for p in sorted(SNAP.iterdir()):
            cfg = p / "config.json"
            if cfg.exists():
                return cfg
    raise SystemExit("checkpoint config.json not found")


def main() -> int:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_gaudi.models.qwen4exp.model import (
        Qwen4ExpDecoderLayer,
        Qwen4ExpForCausalLM,
        Qwen4ExpModel,
    )
    from vllm_gaudi.models.qwen4exp.ple_layer import Qwen4ExpPLELayer

    root = Qwen4ExpConfig.from_json(find_config())
    cfg = root.text

    from vllm.config import ModelConfig

    # Local ModelConfig is a pydantic dataclass that resolves hf_config from a
    # model path; feed it a minimal local llama config, then swap in the shim.
    import json

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
        model=str(dummy),
        tokenizer=str(dummy),
        dtype="bfloat16",
        seed=0,
        max_model_len=64,
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    # Regenerate the architecture-derived config (num_experts, layers, ...)
    # against the swapped-in shim so MoE groups initialize.
    model_config.model_arch_config = model_config.get_model_arch_config()
    assert model_config.is_moe, model_config.get_num_experts()
    vllm_config = VllmConfig(model_config=model_config)

    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp}",
        local_rank=0,
        backend="gloo",
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)

        # The GDN layer pins init-time buffers to the platform device; no HPU
        # driver on this host, so pin everything to CPU for the graph check.
        import vllm.platforms as _vp

        _vp.current_platform.current_device = lambda: torch.device("cpu")  # type: ignore[method-assign]
        model = Qwen4ExpForCausalLM(vllm_config=vllm_config, prefix="")

        layers = list(model.model.layers)
        assert len(layers) == cfg.num_hidden_layers == 48, len(layers)

        from vllm_gaudi.models.qwen4exp.qsa import Qwen4ExpQSAAttention
        from vllm_gaudi.models.qwen4exp.model import Qwen4ExpSparseMoeBlock

        n_gdn = sum(1 for L in layers if hasattr(L, "linear_attn"))
        n_full = sum(1 for L in layers if hasattr(L, "self_attn"))
        assert n_gdn == 36, n_gdn
        assert n_full == 12, n_full
        assert all(isinstance(L.self_attn, Qwen4ExpQSAAttention) for L in layers if hasattr(L, "self_attn"))
        indexer = layers[3].self_attn.indexer
        assert indexer.index_qk_proj.weight.shape == (
            (cfg.indexer_n_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim,
            cfg.hidden_size,
        ), indexer.index_qk_proj.weight.shape
        assert indexer.q_layernorm.weight.shape == (cfg.indexer_head_dim,)
        assert indexer.k_layernorm.weight.shape == (cfg.indexer_head_dim,)

        # PLE lives on the layer whose idx+1 is in ple_layer_ids ([2] -> idx 1)
        ple_layers = [i for i, L in enumerate(layers) if L.ple is not None]
        assert ple_layers == [1], ple_layers
        assert isinstance(layers[1].ple, Qwen4ExpPLELayer)

        # Every decoder layer carries the MoE block + two HC modules.
        for i, L in enumerate(layers):
            assert isinstance(L.mlp, Qwen4ExpSparseMoeBlock), (i, type(L.mlp))
            # Routed experts stay packed: dense BF16 would be ~225 GiB.
            assert not [n for n, _ in L.mlp.experts.named_parameters()], i
            assert L.attn_hyper_connection.hc_count == cfg.hc_count
            assert L.mlp_hyper_connection.hc_lowrank == cfg.hc_lowrank

        moe = layers[0].mlp
        assert moe.n_logical_experts == cfg.num_experts == 512
        assert model.num_routed_experts == cfg.num_experts
        assert model.num_moe_layers == 48

        # State-shape protocol (GDN + PLE short-conv).
        gdn_shape = Qwen4ExpForCausalLM.get_gdn_mamba_state_shape_from_config(vllm_config)
        ple_shape = Qwen4ExpForCausalLM.get_ple_mamba_state_shape_from_config(vllm_config)
        # Vendor format: (conv state, temporal state). Conv covers the fused
        # in_proj conv columns: k_dim*k_heads*2 + v_dim*v_heads.
        gdn_conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        assert gdn_shape[0] == (cfg.linear_conv_kernel_dim - 1, gdn_conv_dim), gdn_shape
        assert gdn_shape[1] == (cfg.linear_num_value_heads,
                                cfg.linear_value_head_dim,
                                cfg.linear_key_head_dim), gdn_shape
        conv_state_len = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size
        hc_hidden = cfg.hidden_size * cfg.hc_count
        assert ple_shape[0] == (conv_state_len, hc_hidden), ple_shape

        # HC merged linear geometry on a real layer (down 320 + inject 4 + pad 12).
        merged = layers[0].attn_hyper_connection.input_mix_weight_down_block_inject
        assert merged.weight.shape[0] == 320 + 4 + 12, merged.weight.shape
        assert layers[0].attn_hyper_connection.input_mix_weight_up.weight.shape == (
            cfg.hc_count * cfg.hidden_size,
            cfg.hc_lowrank,
        )

        # Final mixer: mix-only HC (no inject column).
        final = model.model.hyper_connection_mixer
        assert final.use_combine is False
        assert final.input_mix_weight_down.weight.shape == (320, cfg.hc_count * cfg.hidden_size)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  constructed Qwen4ExpForCausalLM: {n_params/1e6:.1f}M params (BF16 skeleton, no weights)")
        print(f"  layers: 36 GDN + 12 full-attention, PLE at {ple_layers}, 512 experts x 48")
        print(f"  GDN state shapes: {gdn_shape}")
        print(f"  PLE state shape:  {ple_shape}")
        print("test_model_skeleton: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
