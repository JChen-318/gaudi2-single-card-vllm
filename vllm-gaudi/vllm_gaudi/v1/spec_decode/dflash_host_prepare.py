# SPDX-License-Identifier: Apache-2.0
"""Bounded host preparation without changing device math, KV ownership, or RNG."""
import json
import logging

import numpy as np
import torch

from vllm_gaudi.utils import async_h2d_copy

PACK_LIMIT = 1024 * 1024
logger = logging.getLogger(__name__)


class DFlashHostControl:
    """Read bounded local switches only at a changed request-ID tuple."""
    FEATURES = ('handoff', 'sampling_cache', 'draft_inputs', 'packed_copies')

    def __init__(self, path, enabled):
        self.path = path
        self.enabled = bool(enabled)
        self.requests = None
        self.active = dict.fromkeys(self.FEATURES, self.enabled)

    def refresh(self, request_ids):
        requests = tuple(request_ids)
        if requests == self.requests:
            return self.active
        self.requests = requests
        if not self.path or not self.enabled:
            return self.active
        try:
            with open(self.path, encoding='utf-8') as stream:
                raw = stream.read(4097)
            value = json.loads(raw)
            if (len(raw) > 4096 or not isinstance(value, dict) or set(value) != set(self.FEATURES)
                    or any(type(flag) is not bool for flag in value.values())):
                raise ValueError('Invalid bounded DFlash software control')
            self.active = value
        except (OSError, ValueError, TypeError):
            self.active = dict.fromkeys(self.FEATURES, False)
        logger.info('DFlash host software selection: %s', self.active)
        return self.active


def resolved_cpu_pages(block_table, row, logical_pages, pad_block):
    """Read one CPU table row in bulk; caller must exclude defragmentation."""
    indices = np.asarray(logical_pages, dtype=np.int64)
    pages = np.full(indices.shape, pad_block, dtype=np.int64)
    if 0 <= row < block_table.shape[0]:
        live = (indices >= 0) & (indices < block_table.shape[1])
        pages[live] = block_table.numpy()[row, indices[live]]
    pages[(pages < 0) | (pages == pad_block)] = pad_block
    return pages


def make_cpu_slots(block_table, rows, positions, block_size, pad_block, pad_slot):
    """Exact vector addressing for rectangular, non-defragmented CPU inputs."""
    result = []
    for row, row_positions in zip(rows, positions):
        values = np.asarray(row_positions, dtype=np.int64)
        pages = resolved_cpu_pages(block_table, int(row), values // block_size, pad_block)
        result.append(np.where(pages == pad_block, pad_slot, pages * block_size + values % block_size))
    if not result:
        return torch.empty(0, dtype=torch.int64)
    return torch.from_numpy(np.stack(result))


def copy_query_inputs(cpu_values, device):
    """Group small H2D copies by dtype, with owned storage and a size fallback."""
    tensors = [(key, value) for key, value in cpu_values.items() if isinstance(value, torch.Tensor)]
    if any(tensor.device.type != 'cpu' for _, tensor in tensors):
        raise ValueError('DFlash host submission expects CPU inputs')
    if sum(tensor.numel() * tensor.element_size() for _, tensor in tensors) > PACK_LIMIT:
        return {key: async_h2d_copy(value, device=device) if isinstance(value, torch.Tensor) else value
                for key, value in cpu_values.items()}
    groups = {}
    for key, tensor in tensors:
        groups.setdefault(tensor.dtype, []).append((key, tensor))
    result = dict(cpu_values)
    for entries in groups.values():
        if len(entries) == 1:
            key, tensor = entries[0]
            result[key] = async_h2d_copy(tensor, device=device)
            continue
        packed = torch.cat([tensor.reshape(-1) for _, tensor in entries])
        copied = async_h2d_copy(packed, device=device)
        offset = 0
        for key, tensor in entries:
            result[key] = copied.narrow(0, offset, tensor.numel()).view(tensor.shape)
            offset += tensor.numel()
    return result


def select_dflash_parameters(batch, indices, skip_copy):
    """Own selected scalars; invalidate on request/order/value/buffer changes."""
    flags = (batch.all_greedy, batch.no_top_p, batch.no_top_k)
    if any(index < 0 or index >= len(batch.req_ids) or batch.req_ids[index] is None for index in indices):
        # Sampler warmup installs virtual req_id_to_index entries without a
        # live req_ids list. These rows must keep the uncached original selection.
        batch._dflash_sampling_scalar_cache = None
        return (None if flags[0] else batch.temperature[indices],
                None if flags[1] else batch.top_p[indices],
                None if flags[2] else batch.top_k[indices])
    signature = (tuple(indices), tuple(batch.req_ids[index] for index in indices), flags,
                 id(batch.temperature), id(batch.top_p), id(batch.top_k),
                 tuple((float(batch.temperature_cpu[index]), float(batch.top_p_cpu[index]),
                        int(batch.top_k_cpu[index])) for index in indices))
    cached = getattr(batch, '_dflash_sampling_scalar_cache', None)
    if skip_copy and cached is not None and cached[0] == signature:
        return cached[1]
    # Advanced indexing creates private tensors, not aliases of mutable batch buffers.
    selected = (None if flags[0] else batch.temperature[indices],
                None if flags[1] else batch.top_p[indices],
                None if flags[2] else batch.top_k[indices])
    batch._dflash_sampling_scalar_cache = (signature, selected)
    return selected
