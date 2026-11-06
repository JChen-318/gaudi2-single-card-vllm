# SPDX-License-Identifier: Apache-2.0
"""CPU test for the Qwen4Exp checkpoint loader (weights.py).

Validates against the REAL converted checkpoint:
1. NVFP4 / MXFP8 dequant matches the validated numpy reference (csf.py)
   bit-for-bit on actual tensors (routed expert, shared expert, vision MLP).
2. Every checkpoint tensor the trunk model needs maps onto a parameter of the
   constructed skeleton (name coverage via the model's own weight mappers),
   using only the safetensors index — no tensor materialization.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))

CKPT_DIR = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
CKPT = CKPT_DIR / "tensors"


def load_csf_reference():
    spec = importlib.util.spec_from_file_location(
        "csf", REPO / "tools/qwen4exp_flashnext/csf.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    from safetensors.torch import safe_open

    from vllm_gaudi.models.qwen4exp.weights import (
        dequant_mxfp8,
        dequant_nvfp4,
    )

    csf = load_csf_reference()
    weight_map = json.loads((CKPT_DIR / "model.safetensors.index.json").read_text())["weight_map"]

    def get(name: str) -> torch.Tensor:
        with safe_open(CKPT / weight_map[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    # -- 1. dequant correctness on real tensors --------------------------
    base = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    packed, plane, scale2 = (
        get(base + ".weight"),
        get(base + ".weight_scale"),
        get(base + ".weight_scale_2"),
    )
    mine = dequant_nvfp4(packed, plane, scale2)
    ref = csf.dequant_nvfp4(
        packed.numpy(), plane.view(torch.uint8).numpy(), scale2.item()
    )
    assert mine.shape == ref.shape == (640, 2560), (mine.shape, ref.shape)
    exact = torch.equal(mine, torch.from_numpy(ref).to(torch.bfloat16))
    print(f"  expert NVFP4 dequant vs numpy golden: exact={exact}")
    assert exact

    base = "model.language_model.layers.0.mlp.shared_expert.gate_proj"
    weight, scale = get(base + ".weight"), get(base + ".weight_scale")
    mine = dequant_mxfp8(weight, scale)
    ref = csf.dequant_mxfp8(weight.view(torch.uint8).numpy(), scale.numpy())
    exact = torch.equal(mine, torch.from_numpy(ref).to(torch.bfloat16))
    print(f"  shared-expert MXFP8 dequant vs numpy golden: exact={exact}")
    assert exact

    base = "model.visual.blocks.0.mlp.linear_fc1"
    mine = dequant_nvfp4(
        get(base + ".weight"), get(base + ".weight_scale"), get(base + ".weight_scale_2")
    )
    ref = csf.dequant_nvfp4(
        get(base + ".weight").numpy(),
        get(base + ".weight_scale").view(torch.uint8).numpy(),
        get(base + ".weight_scale_2").item(),
    )
    exact = torch.equal(mine, torch.from_numpy(ref).to(torch.bfloat16))
    print(f"  vision MLP NVFP4 dequant: exact={exact} shape={tuple(mine.shape)}")
    assert exact and mine.shape == (4304, 1152)

    # -- 2. name coverage against the model parameter graph --------------
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpConfig
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpForCausalLM

    snap = sorted(
        (
            Path.home()
            / ".cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots"
        ).iterdir()
    )[0]
    cfg = Qwen4ExpConfig.from_json(snap / "config.json").text

    dummy = Path(tempfile.mkdtemp()) / "dummy-model"
    dummy.mkdir(parents=True, exist_ok=True)
    (dummy / "config.json").write_text(json.dumps({
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
        "num_hidden_layers": 1, "num_key_value_heads": 2, "vocab_size": 32,
        "torch_dtype": "bfloat16",
    }))
    model_config = ModelConfig(
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=64
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()

    tmp = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
        local_rank=0, backend="gloo",
    )
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
        import vllm.platforms as vp

        vp.current_platform.current_device = lambda: torch.device("cpu")
        model = Qwen4ExpForCausalLM(vllm_config=vllm_config, prefix="")

        mapper = model.hf_to_vllm_mapper | model.model.hf_to_vllm_mapper

        # Names the trunk load consumes (checkpoint-side filter), keeping the
        # full "model.language_model." prefix so the ForCausalLM mapper can
        # rewrite it to the module's "model." namespace.
        ckpt_names = [
            n for n in weight_map
            if not n.startswith(("model.visual.", "mtp."))
            and "ple.ple_embedding." not in n
            and not n.endswith((".weight_scale", ".weight_scale_2"))
            and n != "model.language_model.hyper_connection_mixer.block_inject_weight.weight"
        ]

        model_params = dict(model.state_dict())

        # Suffixes the loader deliberately ignores (reference parity). The
        # checkpoint stores expert activation-quant inputscales as
        # ``.input_scale`` (dot-separated) — inference never consumes them.
        ignored_suffixes = (
            ".bias", "_bias", ".k_scale", "_k_scale", ".v_scale", "_v_scale",
            "_weight_scale", "_input_scale", ".input_scale",
        )
        # Per-expert names route through the FusedMoE weight_loader instead of
        # state_dict membership; counted separately below.
        def expert_name(n: str) -> bool:
            return ".mlp.experts." in n and n.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"))

        missing, mapped, expert_count = [], set(), 0
        for name in ckpt_names:
            if name.endswith(ignored_suffixes):
                continue
            new = mapper._map_name(name)
            if new is None:
                continue  # mapper drops it deliberately
            if expert_name(name):
                expert_count += 1
                continue
            if new in model_params:
                mapped.add(new)
            else:
                missing.append((name, new))
        print(f"  checkpoint tensors mapped: {len(mapped)} params + {expert_count} expert shards")
        print(f"    (expect 73728 expert shards: 48 layers x 512 experts x 3 projections)")
        for name, new in missing[:12]:
            print(f"    MISSING {name} -> {new}")
        assert not missing, f"{len(missing)} checkpoint names have no parameter"
        assert expert_count == 48 * 512 * 3, expert_count

        # The vendor MoE block passes its shared_expert_gate module into the
        # shared-expert MLP, so the same Parameter appears under two state_dict
        # paths; verify object identity instead of treating the alias as unfed.
        block = model.model.layers[0].mlp
        assert (
            block.shared_expert.expert_gate.weight is block.shared_expert_gate.weight
        ), "shared_expert.expert_gate must alias shared_expert_gate"
        unfed = [
            p for p in model_params
            if p not in mapped
            and not p.endswith("hc_norm.weight")  # Gemma norms: ckpt gives w, model uses (1 + w)
            and ".bias" not in p
            and "_input_mix_padding" not in p
            and "A_log" not in p
            and "dt_bias" not in p
            and ".experts." not in p  # covered by the expert shard count above
            and not p.endswith("shared_expert.expert_gate.weight")  # aliased (identity above)
            and not p.endswith(("_k_scale", "_v_scale", "_q_scale", "_prob_scale"))  # runtime KV-cache scales
        ]
        print(f"  model parameters without checkpoint input: {len(unfed)}")
        for p in unfed[:12]:
            print(f"    UNFED {p}")
        assert len(unfed) == 0, f"{len(unfed)} parameters would stay uninitialized"

    print("test_weights_loader: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
