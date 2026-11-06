# SPDX-License-Identifier: Apache-2.0
"""Bring-up check: the packed-expert MoE block, real weights, on the Gaudi2 HPU.

The vendor block allocates dense BF16 routed experts (225 GiB for the whole
model). ``Qwen4ExpSparseMoeBlock`` keeps the planes packed instead. This
builds one block, loads layer 0's real tensors through its ``load_weights``
(the path the engine uses), and checks the forward against the host.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_moe_block_hpu.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
# The host already points GC_KERNEL_PATH at the stock TPC library, which
# does not carry this GUID. Overwrite rather than setdefault.
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.0.mlp"
_DIST_READY = False


def _weight_map() -> dict[str, str]:
    return json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]


_HANDLES: dict = {}


def _load(name: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = CHECKPOINT / "tensors" / _weight_map()[name]
    if shard not in _HANDLES:
        _HANDLES[shard] = safe_open(shard, framework="pt", device="cpu")
    return _HANDLES[shard].get_tensor(name)


def _build():
    import tempfile

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig
    from vllm_gaudi.models.qwen4exp.model import Qwen4ExpSparseMoeBlock

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

    global _DIST_READY
    if not _DIST_READY:
        tmp = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{tmp}",
            local_rank=0, backend="gloo",
        )
        kept = set_current_vllm_config(vllm_config)
        kept.__enter__()
        initialize_model_parallel(1, 1)
        _DIST_READY = True

    block = Qwen4ExpSparseMoeBlock(vllm_config, prefix="model.layers.0.mlp")
    return block, cfg


def _weights(cfg) -> list[tuple[str, torch.Tensor]]:
    """Block-relative names, as the mapper would hand them to load_weights."""
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    out: list[tuple[str, torch.Tensor]] = [
        ("gate.weight", _load(f"{LAYER}.gate.weight").contiguous()),
        ("shared_expert_gate.weight", _load(f"{LAYER}.shared_expert_gate.weight").contiguous()),
        ("shared_expert.down_proj.weight", _load(f"{LAYER}.shared_expert.down_proj.weight").contiguous()),
    ]
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
            scale2 = base + "_scale_2"
            out.append((f"experts.{expert}.{kind}.weight_scale_2", _load(scale2)))
    return out


def _assert_close(name: str, hpu: torch.Tensor, cpu: torch.Tensor, rel_limit: float) -> None:
    """Relative check: the shared expert reduces over 640 terms at ~2.7e2
    scale, so two independent bf16 roundings sit ~1% apart while each is
    ~0.6% from an fp32 oracle. An absolute limit is meaningless here."""
    abs_max = (hpu.float().cpu() - cpu.float().cpu()).abs().max().item()
    scale = cpu.float().abs().max().item()
    rel = abs_max / max(scale, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        hpu.float().cpu().reshape(1, -1), cpu.float().reshape(1, -1),
    ).item()
    print(f"  {name}: absmax {abs_max:.3e} (scale {scale:.3e}, rel {rel:.3e}, cosine {cosine:.6f})")
    assert torch.isfinite(hpu.float()).all(), name
    assert rel < rel_limit, f"{name}: relative {rel}"
    assert cosine > 0.999, f"{name}: cosine {cosine}"


def main() -> int:
    assert KERNEL.is_file(), KERNEL
    from vllm_gaudi.models.qwen4exp.packed_moe import PackedRoutedExperts

    if not hasattr(torch.ops.custom_op, "custom_qwen4exp_nvfp4_indexed_dequant_bf16_gaudi2"):
        extension = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
        torch.ops.load_library(str(extension))

    block, cfg = _build()
    _, unexpected = block.load_state_dict({}, strict=False)
    assert not [n for n in unexpected if "w13" in n or "w2_" in n], unexpected
    assert isinstance(block.experts, PackedRoutedExperts)
    dense = [n for n, _ in block.named_parameters() if ".experts." in n]
    assert not dense, f"routed experts must not be dense parameters: {dense[:4]}"
    print("  block: routed experts are packed buffers, no dense BF16 w13/w2")

    block.load_weights(iter(_weights(cfg)))
    packed = block.experts.gate_proj_packed
    assert packed.shape[0] == cfg.num_experts, packed.shape
    assert packed.dtype == torch.uint8
    assert block.experts.gate_proj_scale2.abs().sum() > 0, "scale2 was not loaded"
    print(f"  loaded: gate packed {tuple(packed.shape)} uint8, "
          f"scale2 mean {block.experts.gate_proj_scale2.mean().item():.3e}")

    gen = torch.Generator().manual_seed(2026)
    hidden = torch.randn(8, cfg.hidden_size, generator=gen).bfloat16() * 0.5

    cpu_block = block.to("cpu").to(dtype=torch.bfloat16)
    cpu_out = cpu_block(hidden)
    hpu_block = block.to("hpu")
    hpu_out = hpu_block(hidden.to("hpu"))
    torch.hpu.synchronize()
    assert hpu_out.shape == hidden.shape, hpu_out.shape
    _assert_close("MoE block", hpu_out, cpu_out, rel_limit=2e-2)
    print("all packed-expert MoE block HPU tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
