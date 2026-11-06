# SPDX-License-Identifier: Apache-2.0
"""Lazy-mode materialization for Qwen4Exp custom-op inputs.

In lazy mode (``PT_HPU_LAZY_MODE=1``) a *view* tensor -- a split/chunk
slice, a reshape, or the output of a preceding op in an earlier graph
segment -- can reach a custom (TPC) op as a graph input whose
``tensor_data`` is empty: the graph dies at execution with
``ValidateSyncInputTensors ... Strided Params Has Value: 1``.
``.contiguous()`` alone does not fix it, because in lazy mode it can
remain a strided view of its producer. Copying into a fresh buffer
gives the input real storage of its own in both modes (the same
workaround as ``packed_moe.materialized_ids`` and
``hyperconnection_amd._gamma2d``, generalized to any tensor).

Eager mode keeps the plain ``.contiguous()``: no extra copy is added
to the measured eager path. The lazy copy is small (activations, not
weights) against the GEMMs around it.
"""
from __future__ import annotations

import os

import torch

__all__ = ["in_compile_only_mode", "is_hpu_lazy_mode", "lazy_mark_step", "materialized"]


def in_compile_only_mode() -> bool:
    """True under ``PT_COMPILE_ONLY_MODE`` (warmup recipe compilation).

    Synapse compiles recipes without executing tensors there, so a tensor
    created in that mode has no data (``tensor_data`` is empty). Callers
    must not *cache* such a tensor for later real forwards -- the first
    genuinely executed graph that receives it as an input dies with
    ``ValidateSyncInputTensors tensor_data is empty``. The engine enters
    the mode through ``bridge_config.env_setting`` (which does not touch
    ``os.environ``), so the bridge flag is the authoritative probe; the
    env var is also checked for test/launch simulation. Every probe is
    guarded so CPU / non-Habana paths simply return the env result.
    """
    if os.environ.get("PT_COMPILE_ONLY_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    try:
        from habana_frameworks.torch.internal import bridge_config

        return bool(bridge_config.get_pt_compile_only_mode())
    except Exception:
        return False


def is_hpu_lazy_mode() -> bool:
    return os.environ.get("PT_HPU_LAZY_MODE", "").strip().lower() in (
        "1", "true", "yes", "on")


def lazy_mark_step() -> None:
    """End the current lazy graph (no-op in eager mode / off HPU).

    Needed before a host-side read (``int(tensor)``, ``.tolist()``) that
    would otherwise force a partial execution: ops already enqueued but
    not ancestral to the read (e.g. expert dequant before a bucket-cap
    read) can be left unexecuted, and their outputs then reach the next
    graph as empty inputs. Marking first executes them in queue order.
    """
    if not is_hpu_lazy_mode():
        return
    try:
        import habana_frameworks.torch.core as _htcore

        _htcore.mark_step()
    except Exception:
        pass


def materialized(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous tensor with its own storage (fresh copy in lazy mode)."""
    tensor = tensor.contiguous()
    if not is_hpu_lazy_mode():
        return tensor
    out = torch.empty_like(tensor)
    out.copy_(tensor)
    # End the graph at the copy. Without this the copy is just another
    # lazy op: the graph compiler can fuse it into the producer/consumer
    # segment, the result never gets storage of its own at segment
    # boundaries, and the fused segment can die at compile time with
    # synStatus 26 (validateBPGraph) when a later host read or step
    # boundary forces its execution. Marking executes the copy in queue
    # order so the consumer op receives a real tensor. No-op eager.
    lazy_mark_step()
    return out
