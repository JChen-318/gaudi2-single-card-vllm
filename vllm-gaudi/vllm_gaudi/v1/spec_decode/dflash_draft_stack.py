# SPDX-License-Identifier: Apache-2.0
"""DFlash2 draft model as one compiled region (VLLM_HPU_DFLASH2_DRAFT_STACK).

Regional compilation submits the draft embedding, each of the five decoder
layers and the final norm as separate regions. The draft layers are small
(about 0.3 ms of device time each), so every region boundary leaves the device
idle while the host looks up and launches the next one. This module runs the
same upstream modules, in the same order, inside a single region.
"""

from __future__ import annotations

import types

import torch


class HpuDFlashDraftStack(torch.nn.Module):

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        # The draft model keeps sole ownership of its submodules.
        object.__setattr__(self, "_inner", inner)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        inner = self._inner
        hidden_states = inner.embed_input_ids(input_ids)
        residual = None
        for layer in inner.layers:
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
        hidden_states, _ = inner.norm(hidden_states, residual)
        return hidden_states


def _find_draft_model(module: torch.nn.Module) -> torch.nn.Module | None:
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    for candidate in module.modules():
        if isinstance(candidate, DFlashQwen3Model):
            return candidate
    return None


def install_draft_stack(draft_root: torch.nn.Module, compile_fn) -> torch.nn.Module:
    """Route ``DFlashQwen3Model.forward`` without precomputed embeddings to one compiled stack."""
    inner = _find_draft_model(draft_root)
    if inner is None:
        raise RuntimeError("VLLM_HPU_DFLASH2_DRAFT_STACK requires a DFlash Qwen3 draft model")
    stack = compile_fn(HpuDFlashDraftStack(inner))
    original_forward = inner.forward

    def forward(self, input_ids, positions, input_embeds=None):
        if input_embeds is None:
            return stack(input_ids, positions)
        return original_forward(input_ids, positions, input_embeds)

    inner.forward = types.MethodType(forward, inner)
    object.__setattr__(inner, "_hpu_draft_stack", stack)
    return stack
