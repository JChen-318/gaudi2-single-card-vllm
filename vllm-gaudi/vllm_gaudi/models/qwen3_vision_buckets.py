# SPDX-License-Identifier: Apache-2.0
"""Bucketed vision tower for Qwen3-VL style encoders (Qwen3.5) on HPU (VLLM_HPU_VISION_BUCKETS).

The tower runs at each image's native resolution, so every new patch count is a new set of shapes for all 27
blocks: tens of seconds of compilation on the request path the first time a size appears. Here each image is
padded to the next of a few fixed patch counts: the padded patches get zero input, an identity rotation and a
block-diagonal mask (real patches attend only to real ones, padding only to padding), and the merged output is
cut back to the image's real token count. The language model sees exactly the tokens it saw before; only the
encoder's shapes are fixed, and the runner's multimodal warm-up (VLLM_MULTIMODAL_RESOLUTIONS) compiles them.

Images larger than the largest bucket and videos (more than one frame) keep the original path.
"""
from __future__ import annotations

import bisect
import os
from typing import Optional

import torch
import torch.nn.functional as F

_ACTIVE = 0  # bucket size while a bucketed image runs (0 = not running)
_MASK: Optional[torch.Tensor] = None  # its [1, 1, bucket, bucket] mask; None when the image fills the bucket
_ORIGINAL = None


def buckets() -> list[int]:
    raw = os.environ.get("VLLM_HPU_VISION_BUCKETS", "")
    return sorted({int(x) for x in raw.replace(" ", "").split(",") if x})


def current(q_len: int) -> tuple[bool, Optional[torch.Tensor]]:
    """(bucketed image running with this length, its mask or None), read by HpuMMEncoderAttention."""
    return _ACTIVE == q_len and _ACTIVE > 0, _MASK


def bucket_for(patches: int, sizes: list[int]) -> int:
    i = bisect.bisect_left(sizes, patches)
    return sizes[i] if i < len(sizes) else patches


def padding_mask(real: int, size: int, device) -> torch.Tensor:
    is_real = torch.arange(size, device=device) < real
    return (is_real.unsqueeze(0) == is_real.unsqueeze(1)).view(1, 1, size, size)


def _encode_one(tower, pixels: torch.Tensor, grid: list[int], size: int) -> torch.Tensor:
    global _ACTIVE, _MASK
    patches = grid[0] * grid[1] * grid[2]
    meta = tower.prepare_encoder_metadata([grid])
    hidden = tower.patch_embed(pixels.to(device=tower.device, dtype=tower.dtype)) + meta["pos_embeds"]
    cos, sin = meta["rotary_pos_emb_cos"], meta["rotary_pos_emb_sin"]
    pad = size - patches
    if pad:
        hidden = F.pad(hidden, (0, 0, 0, pad))
        cos = F.pad(cos, (0, 0, 0, pad), value=1.0)
        sin = F.pad(sin, (0, 0, 0, pad), value=0.0)
    hidden = hidden.unsqueeze(1)
    _ACTIVE, _MASK = size, (padding_mask(patches, size, hidden.device) if pad else None)
    try:
        deepstack = []
        for layer, block in enumerate(tower.blocks):
            hidden = block(hidden, cu_seqlens=None, rotary_pos_emb_cos=cos, rotary_pos_emb_sin=sin,
                           max_seqlen=meta["max_seqlen"], sequence_lengths=None)
            if layer in tower.deepstack_visual_indexes:
                merger = tower.deepstack_merger_list[tower.deepstack_visual_indexes.index(layer)]
                deepstack.append(merger(hidden))
    finally:
        _ACTIVE, _MASK = 0, None
    out = torch.cat([tower.merger(hidden)] + deepstack, dim=1)
    return out[:patches // (tower.spatial_merge_size**2)]


def bucketed_forward(self, x: torch.Tensor, grid_thw, *, encoder_metadata=None) -> torch.Tensor:
    grid_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
    sizes = buckets()
    if encoder_metadata is not None or not sizes or any(int(g[0]) != 1 for g in grid_list):
        return _ORIGINAL(self, x, grid_thw, encoder_metadata=encoder_metadata)
    outputs, offset = [], 0
    for grid in grid_list:
        grid = [int(v) for v in grid]
        patches = grid[1] * grid[2]
        outputs.append(_encode_one(self, x[offset:offset + patches], grid, bucket_for(patches, sizes)))
        offset += patches
    return torch.cat(outputs, dim=0)


def install() -> bool:
    """Patch the upstream Qwen3 vision tower once, when VLLM_HPU_VISION_BUCKETS is set."""
    global _ORIGINAL
    if _ORIGINAL is not None or not buckets():
        return _ORIGINAL is not None
    from vllm.model_executor.models import qwen3_vl
    _ORIGINAL = qwen3_vl.Qwen3_VisionTransformer.forward
    qwen3_vl.Qwen3_VisionTransformer.forward = bucketed_forward
    return True
