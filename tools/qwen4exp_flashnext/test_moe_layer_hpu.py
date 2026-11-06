# SPDX-License-Identifier: Apache-2.0
"""Bring-up check: packed NVFP4 routed experts on the Gaudi2 HPU.

BF16 expert weights for 48 layers are about 225 GiB and cannot live in
96 GiB HBM. This loads real layer-0 planes, routes a few tokens with the
real gate, dequantizes only the selected experts through the TPC kernel,
and checks the SwiGLU mix against the same host math.

Run:
  PT_HPU_LAZY_MODE=0 python tools/qwen4exp_flashnext/test_moe_layer_hpu.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
# The host often already points GC_KERNEL_PATH at the stock TPC library.
# setdefault would keep that and the custom GUID would not resolve.
os.environ["GC_KERNEL_PATH"] = str(KERNEL)
os.environ.setdefault("PT_HPU_LAZY_MODE", "0")

sys.path.insert(0, str(REPO / "vllm-gaudi"))

import torch  # noqa: E402

CHECKPOINT = Path(os.environ.get("QWEN4EXP_MODEL", "~/models/qwen4exp-flashnext-plain")).expanduser()
LAYER = "model.language_model.layers.0.mlp"


def _weight_map() -> dict[str, str]:
    index = CHECKPOINT / "model.safetensors.index.json"
    return json.loads(index.read_text())["weight_map"]


def _load(weight_map: dict[str, str], name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(CHECKPOINT / "tensors" / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _expert_planes(weight_map: dict[str, str], experts: list[int], kind: str):
    packed, scales, scale2 = [], [], []
    for expert in experts:
        base = f"{LAYER}.experts.{expert}.{kind}.weight"
        plane = _load(weight_map, base + "_scale").view(torch.uint8).contiguous()
        packed.append(_load(weight_map, base).contiguous())
        scales.append(plane)
        scale_name = base + "_scale_2"
        scale2.append(float(_load(weight_map, scale_name).reshape(-1)[0]) if scale_name in weight_map else 1.0)
    return (
        torch.stack(packed),
        torch.stack(scales),
        torch.tensor(scale2, dtype=torch.float32),
    )


def _assert_close(name: str, hpu: torch.Tensor, cpu: torch.Tensor, abs_limit: float) -> None:
    diff = (hpu.float().cpu() - cpu.float()).abs()
    abs_max = diff.max().item()
    cosine = torch.nn.functional.cosine_similarity(
        hpu.float().cpu().reshape(1, -1), cpu.float().reshape(1, -1),
    ).item()
    print(f"  {name}: absmax {abs_max:.3e} cosine {cosine:.6f}")
    assert torch.isfinite(hpu.float()).all(), name
    assert abs_max < abs_limit, f"{name}: absmax {abs_max}"
    assert cosine > 0.999, f"{name}: cosine {cosine}"


def main() -> int:
    assert KERNEL.is_file(), KERNEL
    # Platform registration before the custom-op library. The other order
    # leaves the first TPC graph with an empty synapse graph (synStatus 26).
    from vllm_gaudi.models.qwen4exp.packed_moe import (
        apply_routed_experts,
        dequant_indexed,
        route_topk,
    )

    extension = next((REPO / "vllm-gaudi/csrc/qwen4exp/pytorch").glob("hpu_qwen4exp_nvfp4_dequant_pt2*.so"))
    torch.ops.load_library(str(extension))

    weight_map = _weight_map()
    gate = _load(weight_map, f"{LAYER}.gate.weight").contiguous()
    gen = torch.Generator().manual_seed(2026)
    hidden = torch.randn(4, gate.shape[1], generator=gen).bfloat16() * 0.5
    topk_ids, topk_weights = route_topk(hidden, gate, top_k=10)
    selected = sorted({int(expert) for expert in topk_ids.reshape(-1).tolist()})
    compact = {expert: index for index, expert in enumerate(selected)}
    local_ids = torch.tensor(
        [[compact[int(expert)] for expert in row] for row in topk_ids.tolist()],
        dtype=torch.int32,
    )
    print(f"  routed {topk_ids.shape[-1]} of {gate.shape[0]}; unique experts {len(selected)}")

    planes = {kind: _expert_planes(weight_map, selected, kind) for kind in ("gate_proj", "up_proj", "down_proj")}
    print(f"  scale2 gate {planes['gate_proj'][2][:4].tolist()}")

    def project(device: str):
        # CPU ids: an HPU->CPU copy fused with the weight upload is the
        # empty-graph compile failure. Upload, sync, then dequant.
        ids = torch.arange(len(selected), dtype=torch.int32)

        def weights(kind: str) -> torch.Tensor:
            packed, scales, scale2 = planes[kind]
            packed_d, scales_d = packed.to(device), scales.to(device)
            if device == "hpu":
                torch.hpu.synchronize()
            return dequant_indexed(packed_d, scales_d, scale2, ids)

        gate_w, up_w, down_w = weights("gate_proj"), weights("up_proj"), weights("down_proj")
        if device == "hpu":
            torch.hpu.synchronize()
        return gate_w, up_w, down_w

    hpu_gate, hpu_up, hpu_down = project("hpu")
    cpu_gate, cpu_up, cpu_down = project("cpu")
    _assert_close("gate dequant", hpu_gate, cpu_gate, abs_limit=1e-3)
    _assert_close("up dequant", hpu_up, cpu_up, abs_limit=1e-3)
    _assert_close("down dequant", hpu_down, cpu_down, abs_limit=1e-3)

    hpu_out = apply_routed_experts(
        hidden.to("hpu"), local_ids.to("hpu"), topk_weights.to("hpu"), hpu_gate, hpu_up, hpu_down,
    )
    torch.hpu.synchronize()
    cpu_out = apply_routed_experts(hidden, local_ids, topk_weights, cpu_gate, cpu_up, cpu_down)
    assert hpu_out.shape == (4, 2560), hpu_out.shape
    _assert_close("routed mix", hpu_out, cpu_out, abs_limit=2e-2)
    print("all packed MoE HPU tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
