# SPDX-License-Identifier: Apache-2.0
"""Bring-up check: real-weight GDN layer forward on the Gaudi2 HPU.

Loads ``model.layers.0.linear_attn`` from the converted checkpoint (MXFP8
projections dequantized to BF16), runs two prefill chunks through
``HPUGatedDeltaNetAttention`` with a minimal ``HPUAttentionMetadataV1``
stand-in, and checks:

1. The vllm-gaudi monkeypatch is what the Qwen4Exp decoder constructs.
2. Chunk 2 with carried conv/ssm state differs from a fresh-state rerun, so
   the metadata fixture actually feeds the vendor kernels.
3. HPU BF16 matches a CPU BF16 run of the same layer within bf16 tolerance.
   An fp32-weight oracle is not used.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_gdn_layer_hpu.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
sys.path.insert(0, str(Path(__file__).parent))

import vllm_gaudi  # noqa: F401, E402  (registers the HPU platform first)

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.0.linear_attn"
_DIST_READY = False
_VLLM_CONFIG = None


def _shards():
    return sorted((CHECKPOINT / "tensors").glob("*.safetensors"))


def _headers():
    for shard in _shards():
        with open(shard, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            yield shard, json.loads(f.read(size))


def _load(name: str) -> torch.Tensor:
    from safetensors import safe_open

    for shard, header in _headers():
        if name in header:
            with safe_open(shard, framework="pt", device="cpu") as f:
                return f.get_tensor(name)
    raise FileNotFoundError(name)


def _dequant_mxfp8(base: str) -> torch.Tensor:
    from vllm_gaudi.models.qwen4exp.weights import dequant_mxfp8

    weight = _load(f"{base}.weight")
    scale = _load(f"{base}.weight_scale")
    out = dequant_mxfp8(weight, scale)
    assert out.shape == weight.shape, (base, out.shape, weight.shape)
    return out


def _layer_weights() -> dict[str, torch.Tensor]:
    """Checkpoint tensors -> fused GDN state_dict entries (BF16 projections)."""
    qkv = _dequant_mxfp8(f"{LAYER}.in_proj_qkv")
    z = _dequant_mxfp8(f"{LAYER}.in_proj_z")
    b = _dequant_mxfp8(f"{LAYER}.in_proj_b")
    a = _dequant_mxfp8(f"{LAYER}.in_proj_a")
    return {
        "in_proj_qkvz.weight": torch.cat((qkv, z), dim=0).contiguous(),
        "in_proj_ba.weight": torch.cat((b, a), dim=0).contiguous(),
        "conv1d.weight": _load(f"{LAYER}.conv1d.weight").contiguous(),
        "A_log": _load(f"{LAYER}.A_log"),
        "dt_bias": _load(f"{LAYER}.dt_bias"),
        "norm.weight": _load(f"{LAYER}.norm.weight"),
        "out_proj.weight": _dequant_mxfp8(f"{LAYER}.out_proj"),
    }


def _build():
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
    from vllm_gaudi.models.qwen4exp.config import Qwen4ExpTextConfig

    text = json.load(open(CHECKPOINT / "config.json"))["text_config"]
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
        model=str(dummy), tokenizer=str(dummy), dtype="bfloat16", seed=0, max_model_len=8192
    )
    model_config.hf_config = cfg
    model_config.hf_text_config = cfg
    model_config.model_arch_config = model_config.get_model_arch_config()
    vllm_config = VllmConfig(model_config=model_config)
    # Ordinary one-request prefill: do not take the prefix-cache block path.
    vllm_config.cache_config.enable_prefix_caching = False
    global _DIST_READY, _VLLM_CONFIG
    _VLLM_CONFIG = vllm_config
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

    torch.manual_seed(0)
    layer = HPUGatedDeltaNetAttention(
        cfg, vllm_config=vllm_config, prefix="model.layers.0.linear_attn", gqa_interleaved_layout=False,
    )
    state = _layer_weights()
    cast = {}
    for name, weight in state.items():
        param = dict(layer.named_parameters()).get(name)
        cast[name] = weight.to(dtype=param.dtype) if param is not None else weight
    missing, unexpected = layer.load_state_dict(cast, strict=False)
    real_missing = [m for m in missing if not m.endswith(".bias")]
    assert not real_missing, real_missing
    assert not unexpected, unexpected
    # Linear params follow the platform default (fp32) unless the loader sets
    # them. The checkpoint planes are BF16; gating upcasts A_log / dt_bias.
    layer.to(dtype=torch.bfloat16)
    assert isinstance(layer, HPUGatedDeltaNetAttention)
    return layer


def _metadata(num_tokens: int, device: str, has_initial: bool) -> SimpleNamespace:
    """The fields ``HPUGatedDeltaNetAttention._extract_metadata`` reads."""
    index = torch.tensor([0], dtype=torch.int32, device=device)
    return SimpleNamespace(
        is_prompt=True,
        load_indices_tensor=index,
        store_indices_tensor=index,
        query_start_loc_p=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        has_initial_states_p=torch.tensor([has_initial], device=device),
        padding_mask_flat=None,
        direct_gdn_state=False,
        num_accepted_tokens=None,
        gdn_initial_state_indices=None,
        dflash_full_query=False,
    )


def _bind_cache(layer, device: str) -> None:
    conv_shape, ssm_shape = layer.get_state_shape()
    conv_dtype, ssm_dtype = layer.get_state_dtype()
    # Slot 1 is the kernel's garbage slot (remainder of a pad index).
    layer.kv_cache = (
        torch.zeros((2, *conv_shape), dtype=conv_dtype, device=device),
        torch.zeros((2, *ssm_shape), dtype=ssm_dtype, device=device),
    )


def _run(layer, chunks: list[torch.Tensor], device: str):
    from vllm.forward_context import set_forward_context

    _bind_cache(layer, device)
    outs = []
    for i, hidden in enumerate(chunks):
        tokens = hidden.to(device)
        meta = _metadata(tokens.shape[0], device, has_initial=i > 0)
        with set_forward_context(meta, _VLLM_CONFIG, num_tokens=tokens.shape[0]):
            outs.append(layer(tokens))
        if device == "hpu":
            torch.hpu.synchronize()
    conv, ssm = layer.kv_cache
    return outs, (conv.detach().clone(), ssm.detach().clone())


def _assert_close(name: str, hpu: torch.Tensor, cpu: torch.Tensor, abs_limit: float) -> None:
    diff = (hpu.float().cpu() - cpu.float()).abs()
    abs_max = diff.max().item()
    denom = cpu.float().norm().clamp(min=1e-8)
    cosine = torch.nn.functional.cosine_similarity(
        hpu.float().cpu().reshape(1, -1), cpu.float().reshape(1, -1)
    ).item()
    print(f"  {name}: absmax {abs_max:.3e} cosine {cosine:.6f}")
    assert torch.isfinite(hpu.float()).all(), name
    assert abs_max < abs_limit, f"{name}: HPU vs CPU bf16 absmax {abs_max}"
    assert cosine > 0.999, f"{name}: cosine {cosine}"


def main() -> int:
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_mod
    from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
    from vllm_gaudi.models.qwen4exp import model as q4_model

    # Serve order applies the patch before the decoder is constructed. The
    # decoder looks the class up at construction, so an earlier import of
    # model.py must not freeze the vendor class.
    assert gdn_mod.QwenGatedDeltaNetAttention is HPUGatedDeltaNetAttention
    assert q4_model._gdn_mod.QwenGatedDeltaNetAttention is HPUGatedDeltaNetAttention
    print("  GDN class: decoder lookup is HPUGatedDeltaNetAttention")

    layer = _build()
    gen = torch.Generator().manual_seed(2026)
    # One chunk (128) then a second that must read the written conv/ssm state.
    chunks = [
        torch.randn(128, 2560, generator=gen).bfloat16() * 0.5,
        torch.randn(128, 2560, generator=gen).bfloat16() * 0.5,
    ]

    hpu_layer = layer.to("hpu")
    hpu_out, (hpu_conv, hpu_ssm) = _run(hpu_layer, chunks, "hpu")
    for i, out in enumerate(hpu_out):
        assert out.shape == (128, 2560), (i, out.shape)
        assert out.device.type == "hpu"
        assert torch.isfinite(out.float()).all()
    assert hpu_conv[0].abs().sum() > 0, "conv state was not written"
    assert hpu_ssm[0].abs().sum() > 0, "ssm state was not written"
    print("  HPU prefill: two chunks finite, conv/ssm slot 0 written")

    # Same second chunk against an empty cache must not match the carried run.
    fresh = hpu_layer
    _bind_cache(fresh, "hpu")
    from vllm.forward_context import set_forward_context

    with set_forward_context(_metadata(128, "hpu", has_initial=False), _VLLM_CONFIG, num_tokens=128):
        fresh_out = fresh(chunks[1].to("hpu"))
    torch.hpu.synchronize()
    carried_delta = (hpu_out[1].float() - fresh_out.float()).abs().max().item()
    assert carried_delta > 1e-3, f"carried state did not affect chunk 2 ({carried_delta})"
    print(f"  HPU state carry: chunk 2 vs fresh absmax {carried_delta:.3e}")

    cpu_layer = hpu_layer.to("cpu")
    cpu_out, (cpu_conv, cpu_ssm) = _run(cpu_layer, chunks, "cpu")
    # Recurrent bf16 over 256 tokens. Same kernels, so the gap is device rounding.
    # Measured 2026-10-25: output 5.9e-3, conv 3.9e-3, ssm 1.5e-3.
    for i, (h, c) in enumerate(zip(hpu_out, cpu_out)):
        _assert_close(f"chunk {i}", h, c, abs_limit=2e-2)
    _assert_close("conv state", hpu_conv[0], cpu_conv[0], abs_limit=2e-2)
    _assert_close("ssm state", hpu_ssm[0], cpu_ssm[0], abs_limit=2e-2)
    print("all GDN layer HPU tests passed")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
    raise SystemExit(main())
