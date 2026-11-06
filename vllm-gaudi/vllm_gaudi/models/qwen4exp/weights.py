# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint loader for the Qwen4Exp HPU port.

Streams the CSF-converted checkpoint (plain NVFP4 scale planes, see
``tools/qwen4exp_flashnext/convert.py``) and yields BF16 tensors:

- BF16 / F32 / I64 tensors pass through unchanged.
- Routed-expert and vision-MLP NVFP4 groups (``weight`` U8 packed nibbles +
  ``weight_scale`` F8_E4M3 [rows, cols/16] + optional ``weight_scale_2`` F32)
  dequantize exactly (E2M1 x E4M3 products are BF16-representable).
- MXFP8 groups (attention / shared experts: ``weight`` F8_E4M3 +
  ``weight_scale`` U8 [rows, cols/32] UE8M0) dequantize as
  ``e4m3 * 2^(ue8m0 - 127)``.
- PLE n-gram shard tables and MTP weights are skipped here: the tables live
  on pinned host memory (PLE runtime) and the MTP head is ported separately.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import safe_open

from .csf_math import e2m1_value_lut, e4m3_value_lut

__all__ = ["dequant_mxfp8", "dequant_nvfp4", "iter_bf16_weights"]

# Tensors consumed outside the trunk model load.
SKIP_SUBSTRINGS = (
    "ple.ple_embedding.",  # host-resident n-gram shard tables + metadata
    "mtp.",                # speculative head loads separately
)

_E2M1 = e2m1_value_lut()  # [16] float32: (code & 15) -> value (sign in bit 3)
_E4M3 = e4m3_value_lut()  # [256] float32


def dequant_nvfp4(
    packed: torch.Tensor,
    plane: torch.Tensor,
    scale2: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """``packed`` U8 [r, c/2], ``plane`` F8_E4M3 [r, c/16] -> BF16 [r, c]."""
    rows, half = packed.shape
    cols = half * 2
    if plane.shape != (rows, cols // 16):
        raise ValueError(f"plane {tuple(plane.shape)} != ({rows}, {cols // 16})")
    bytes_ = packed.view(torch.uint8).to(torch.int64)
    # Low nibble first, matching the CSF payload layout.
    nibbles = torch.stack([bytes_ & 0xF, (bytes_ >> 4) & 0xF], dim=-1).flatten(-2)
    scale_codes = plane.view(torch.uint8).to(torch.int64)
    scale_vals = _E4M3[scale_codes]
    out = _E2M1[nibbles] * scale_vals.repeat_interleave(16, dim=-1)
    if isinstance(scale2, torch.Tensor):
        if scale2.numel() != 1:
            raise ValueError(f"scale2 must be scalar, got {tuple(scale2.shape)}")
        out = out * scale2.to(torch.float32).reshape(1)
    elif scale2 != 1.0:
        out = out * scale2
    return out.to(torch.bfloat16)


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``weight`` F8_E4M3 [r, K], ``scale`` U8 [r, K/32] -> BF16 [r, K]."""
    rows, cols = weight.shape
    if scale.shape != (rows, cols // 32):
        raise ValueError(f"scale {tuple(scale.shape)} != ({rows}, {cols // 32})")
    codes = weight.view(torch.uint8).to(torch.int64)
    values = _E4M3[codes]
    factor = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
    return (values * factor.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _shard_header(shard: Path) -> dict:
    with open(shard, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def iter_bf16_weights(
    shards: list[Path] | Iterator[Path],
    *,
    skip_substrings: tuple[str, ...] = SKIP_SUBSTRINGS,
    only_substrings: tuple[str, ...] | None = None,
    packed_experts: bool = True,
    packed_expert_substr: str = ".mlp.experts.",
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for the whole converted checkpoint.

    Names keep the checkpoint's ``model.language_model.`` prefix; strip it at
    the call site (the ForCausalLM mapper handles ``model.``-relative names).
    ``only_substrings`` restricts the stream to matching names before any
    dequantization (used by the draft loader to take just the ``mtp.*``
    group without materializing the trunk).

    ``packed_experts`` keeps routed-expert NVFP4 plane groups undequantized
    and emits the weight, the scale plane and ``weight_scale_2`` separately
    so ``Qwen4ExpSparseMoeBlock`` can store them packed. Everything else is
    still dequantized on the fly.
    """
    entries: dict[str, tuple[Path, dict]] = {}
    for shard in shards:
        for name, meta in _shard_header(shard).items():
            if name == "__metadata__":
                continue
            if any(s in name for s in skip_substrings):
                continue
            if only_substrings is not None and not any(s in name for s in only_substrings):
                continue
            entries[name] = (shard, meta)

    planes = {
        n[: -len(".weight_scale")]: n
        for n in entries
        if n.endswith(".weight_scale")
    }
    scale2s = {
        n[: -len(".weight_scale_2")]: n
        for n in entries
        if n.endswith(".weight_scale_2")
    }

    handles: dict[Path, object] = {}

    def tensor(shard: Path, name: str) -> torch.Tensor:
        if shard not in handles:
            handles[shard] = safe_open(shard, framework="pt", device="cpu")
        return handles[shard].get_tensor(name)

    for name, (shard, meta) in entries.items():
        dtype = meta["dtype"]
        if dtype in ("BF16", "F32", "I64", "U16", "I32"):
            yield name, tensor(shard, name)
            continue
        if not name.endswith(".weight") or dtype not in ("U8", "F8_E4M3"):
            # Scale planes are consumed together with their base weight.
            continue
        base = name[: -len(".weight")]
        plane_name = planes.get(base)
        if plane_name is None:
            raise ValueError(f"{name}: quantized tensor without weight_scale")
        weight = tensor(shard, name)
        plane_shard, plane_meta = entries[plane_name]
        plane = tensor(plane_shard, plane_name)
        rows, cols = meta["shape"]
        if plane_meta["shape"] == [rows, cols // 32] and dtype == "F8_E4M3":
            # MXFP8: the weight keeps the full K, so the UE8M0 plane is K/32.
            yield name, dequant_mxfp8(weight, plane)
        elif plane_meta["shape"] == [rows, cols // 8] and dtype == "U8":
            # NVFP4: U8 packs two E2M1 values, so the row is K/2 bytes and
            # the per-16 scale plane holds K/16 = cols/8 entries.
            scale2_name = scale2s.get(base)
            scale2 = (
                tensor(entries[scale2_name][0], scale2_name)
                if scale2_name is not None
                else 1.0
            )
            if packed_experts and packed_expert_substr in name:
                # Dense BF16 experts for 48 layers would be ~225 GiB. Keep
                # the planes packed and hand the scale to the MoE block.
                yield name, weight
                yield plane_name, plane.view(torch.uint8)
                if scale2_name is not None:
                    yield scale2_name, scale2
                continue
            yield name, dequant_nvfp4(weight, plane, scale2)
        else:
            raise ValueError(
                f"{base}: {dtype} weight {meta['shape']} with plane "
                f"{plane_meta['dtype']} {plane_meta['shape']}"
            )
