"""Keep one copy of weights that the fused paths concatenate (H31).

The GDN input projection (H14), the QSA projection (H23-4) and the shared expert (H15) each run as
one GEMM over a concatenation of two weights. The concatenation is cached, and until H31 the two
original parameters stayed resident beside it: 3.9 GiB of duplicate weights on a card where every
GiB is context. :func:`alias_params_into` points each parameter at its rows of the concatenation,
so the original storage is freed and both paths read the same bytes (the values are unchanged).
``VLLM_QWEN4EXP_ALIAS_MERGED=0`` keeps the duplicates.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def alias_enabled() -> bool:
    return os.environ.get("VLLM_QWEN4EXP_ALIAS_MERGED", "1") != "0"


def merge_rows(*params: torch.Tensor) -> torch.Tensor:
    """``torch.cat(params, dim=0)``, then :func:`alias_params_into` it.

    The concatenation is made outside inference mode: the parameters become views of it, and the
    fused paths key their caches on the parameters' version counters, which inference tensors lack.
    """
    if any(param.untyped_storage().nbytes() > param.nbytes for param in params):
        # The parameters are already views of an earlier concatenation: whoever still holds that
        # one (the decode graph's bound arguments) now keeps a duplicate alive.
        logger.warning("re-merging already aliased parameters %s", [tuple(p.shape) for p in params])
    with torch.inference_mode(False), torch.no_grad():
        merged = torch.cat([param.detach() for param in params], dim=0).contiguous()
    alias_params_into(merged, *params)
    return merged


def views_of(merged: torch.Tensor | None, *params: torch.Tensor) -> bool:
    """Whether ``params`` are, in order, the row blocks of ``merged`` (what :func:`alias_params_into` made).

    The fused paths used to validate their cached concatenation by the parameters' ``data_ptr()``.
    Once the parameters are views of the concatenation that check is wrong twice over: the device
    allocator can move the storage (both the cache and the parameters move with it, so the cache is
    still right, but the pointers in the key no longer match), and a re-merge then made a second
    copy while the decode graph's bound arguments kept the first one alive: 1.3 GiB of stale
    duplicates in H31. Storage identity and offsets survive a move, and an in-place update of a
    parameter writes the concatenation itself, so this check is all the cache needs.
    """
    if merged is None or merged.ndim != 2 or not merged.is_contiguous():
        return False
    try:
        base = merged.untyped_storage().data_ptr()
        offset = merged.storage_offset()
        cols = int(merged.shape[1])
        for param in params:
            if (param.ndim != 2 or int(param.shape[1]) != cols or param.dtype != merged.dtype
                    or param.device != merged.device or not param.is_contiguous()
                    or param.untyped_storage().data_ptr() != base or param.storage_offset() != offset):
                return False
            offset += int(param.shape[0]) * cols
    except Exception:  # noqa: BLE001 - tensors without storage (fake / meta) never match
        return False
    return offset == merged.storage_offset() + merged.numel()


def alias_params_into(merged: torch.Tensor, *params: torch.Tensor) -> bool:
    """Make ``params`` (stacked by rows, in order) views of ``merged``. Returns whether it did."""
    if (not alias_enabled() or merged.device.type != "hpu" or merged.ndim != 2 or not merged.is_contiguous()
            or merged.is_inference()):
        return False
    offset = 0
    for param in params:
        if (param.ndim != 2 or param.shape[1] != merged.shape[1] or param.dtype != merged.dtype
                or param.device != merged.device):
            return False
        offset += int(param.shape[0])
    if offset != merged.shape[0]:
        return False
    offset = 0
    with torch.no_grad():
        for param in params:
            rows = int(param.shape[0])
            param.data = merged[offset:offset + rows]
            offset += rows
    return True
