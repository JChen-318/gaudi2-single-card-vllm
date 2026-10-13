# SPDX-License-Identifier: Apache-2.0
"""Opt-in, functional T6 GDN core. No padding or sampling changes.

Load the registration extension once before compiling. The native kernel
produces six states; accepted-state selection and checkpoint writes stay in
the surrounding graph. Production enablement requires HPU qualification.
"""

from __future__ import annotations

import os

import torch

_ENABLED = os.getenv("FLASHINFER_GAUDI_ENABLE_MTP5_T6", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_CORE = None
if _ENABLED:
    _library = os.environ.get("FLASHINFER_GAUDI_MTP5_T6_OP_LIBRARY", "")
    _kernel = os.environ.get("FLASHINFER_GAUDI_MTP5_T6_KERNEL_LIBRARY", "")
    if (
        not _library
        or not _kernel
        or _kernel not in os.getenv("GC_KERNEL_PATH", "").split(":")
    ):
        raise RuntimeError(
            "MTP5 T6 requires explicit registration and GC kernel library paths"
        )
    torch.ops.load_library(_library)
    _CORE = torch.ops.custom_op.flashinfer_gaudi_gdn_mtp5_t6_prepared


def eligible_t6(
    packed,
    log_decay,
    beta,
    state_pool,
    state_indices,
    accepted,
    query_lengths,
    scale,
    normalize,
    full_query,
    policy,
):
    """Shape/proof-only gate; no device-to-host reads in the compiled path.

    Only full-query single-stream T6 is promoted. Shortened verification,
    padding, mixed batches and unsupported policies retain the old dispatcher.
    """
    return (
        _CORE is not None
        and policy == "auto"
        and torch.compiler.is_compiling()
        and full_query
        and normalize
        and packed.device.type in ("hpu", "privateuseone")
        and packed.dtype == torch.bfloat16
        and tuple(packed.shape) == (1, 6, 10240)
        and state_pool.dtype == torch.float32
        and state_pool.ndim == 4
        and tuple(state_pool.shape[1:]) == (48, 128, 128)
        and state_pool.shape[0] > 0
        and log_decay.dtype == torch.float32
        and tuple(log_decay.shape) == (1, 6, 48)
        and beta.dtype == torch.bfloat16
        and beta.shape == log_decay.shape
        and state_indices.dtype == torch.int32
        and tuple(state_indices.shape) == (1, 6)
        and accepted.dtype == torch.int32
        and tuple(accepted.shape) == (1,)
        and query_lengths.dtype == torch.int32
        and tuple(query_lengths.shape) == (1,)
        and all(
            x.device == packed.device
            for x in (
                log_decay,
                beta,
                state_pool,
                state_indices,
                accepted,
                query_lengths,
            )
        )
        and (scale is None or abs(scale - 128**-0.5) <= 1e-12)
    )


def execute_prepared_t6(
    packed,
    log_decay,
    beta,
    state_pool,
    state_indices,
    accepted,
    normalize_fn,
    core=None,
    distinct=False,
    checkpoint_start=None,
    initial_state_indices=None,
):
    """Caller proves every T6 token and checkpoint index is valid.

    Explicit initial-state indices preserve rollback from a checkpoint outside
    the current output width. Repeated destinations retain ordered writes.
    The functional kernel never mutates its input state.
    """
    if tuple(packed.shape) != (1, 6, 10240):
        raise ValueError("The MTP5 prepared path requires full B1/T6")
    if tuple(state_indices.shape) != (1, 6) or tuple(accepted.shape) != (1,):
        raise ValueError("Invalid T6 checkpoint or accepted-token shapes")
    if checkpoint_start is not None and (
        not distinct
        or checkpoint_start < 1
        or checkpoint_start + 6 > state_pool.shape[0]
    ):
        raise ValueError("Direct T6 checkpoints require six valid distinct slots")
    if initial_state_indices is None:
        offsets = accepted.long().clamp(1, 6) - 1
        # B1 int64 gather is unsupported on this runtime; select in int32.
        mask = torch.arange(6, device=state_indices.device)[None, :] == offsets[:, None]
        loads = torch.where(mask, state_indices, 0).sum(1).long()
    else:
        if tuple(initial_state_indices.shape) != (1,):
            raise ValueError("Explicit loads require one checkpoint per request")
        loads = initial_state_indices.long()
    initial_state = state_pool.index_select(0, loads).contiguous()
    qk = normalize_fn(packed[..., :4096].float().reshape(1, 6, 32, 128))
    q = qk[..., :16, :] * (128**-0.5)
    k = qk[..., 16:, :]
    prepared = torch.cat(
        (q.flatten(2), k.flatten(2), packed[..., 4096:].float()), -1
    ).contiguous()
    core = _CORE if core is None else core
    if core is None:
        raise RuntimeError("The T6 registration extension is not loaded")
    output, checkpoints = core(
        initial_state,
        prepared,
        torch.exp(log_decay).contiguous(),
        beta.float().contiguous(),
    )
    if checkpoint_start is not None:
        state_pool.narrow(0, checkpoint_start, 6).copy_(
            checkpoints.reshape(6, 48, 128, 128)
        )
    elif distinct:
        state_pool.index_copy_(
            0, state_indices.flatten().long(), checkpoints.reshape(6, 48, 128, 128)
        )
    else:
        for token in range(6):
            state_pool.index_copy_(
                0, state_indices[:, token].long(), checkpoints[:, token]
            )
    return output.to(packed.dtype), state_pool
