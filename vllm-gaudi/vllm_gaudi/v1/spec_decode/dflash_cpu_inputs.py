# SPDX-License-Identifier: Apache-2.0
"""Owned CPU inputs for the accepted-prefix DFlash proposer.

No cached addresses or HPU operations: request order, accepted lengths, and
mutable page tables are read anew for every invocation.
"""
import numpy as np
import torch
from vllm_gaudi.v1.spec_decode.dflash_host_prepare import (
    DFlashHostControl,
    resolved_cpu_pages,
)


class DFlashDecodeControl(DFlashHostControl):
    """Independent per-request switches for comparison without another restart."""
    FEATURES = ('cpu_inputs', 'context_region')


def prepare_context_cpu_inputs(block_table, rows, starts, counts, target_width,
                               block_size, pad_block, pad_slot):
    """Pack source rows, absolute positions and physical slots in one buffer."""
    offsets = [np.arange(count, dtype=np.int64) for count in counts]
    sources = np.concatenate([row * target_width + offset for row, offset in zip(rows, offsets)])
    positions = np.concatenate([start + offset for start, offset in zip(starts, offsets)])
    slots = []
    for row, start, offset in zip(rows, starts, offsets):
        values = start + offset
        pages = resolved_cpu_pages(block_table, row, values // block_size, pad_block)
        slots.append(np.where(pages == pad_block, pad_slot, pages * block_size + values % block_size))
    # One owned allocation prevents mutation of batch/table buffers while an
    # asynchronous transfer consumes this snapshot. Each row keeps its order.
    return torch.from_numpy(np.stack((sources, positions, np.concatenate(slots))))


def bucket_context_blocks(count: int) -> int:
    """Draft context page buckets: powers of two up to 32 pages, then multiples of 32.

    The draft attention's page list otherwise grows one page at a time, and each
    new length compiles a new graph in the middle of serving. Extra pages hold the
    padding block and their positions are masked by the explicit bias.
    """
    if count <= 0:
        return 0
    if count <= 32:
        return 1 << (count - 1).bit_length()
    return (count + 31) // 32 * 32


def prepare_query_cpu_inputs(block_table, rows, starts, bonus_ids, *, batch_size,
                             query_len, block_size, mask_token_id, pad_block,
                             pad_slot, window, causal, dtype, bucket_blocks=None):
    """Construct the original query and explicit bias with vector CPU work."""
    real_batch = len(rows)
    if not real_batch or len(starts) != real_batch or len(bonus_ids) != real_batch:
        raise ValueError('DFlash query needs matching, nonempty rows, starts and bonus ids')
    if batch_size < real_batch or block_size < 1 or query_len < 1 or (window is not None and window < 1):
        raise ValueError('Invalid DFlash query dimensions')
    starts = np.asarray(list(starts) + [0] * (batch_size - real_batch), dtype=np.int64)
    rows = list(rows) + [-1] * (batch_size - real_batch)
    positions = starts[:, None] + np.arange(query_len, dtype=np.int64)[None, :]
    first = np.zeros(batch_size, dtype=np.int64) if window is None else np.maximum(0, starts - (window - 1)) // block_size
    last = (starts - 1) // block_size
    widths = np.maximum(0, last - first + 1)
    max_blocks = int(widths.max())
    if bucket_blocks is not None:
        max_blocks = bucket_blocks(max_blocks)
    blocks = np.full((batch_size, max_blocks), pad_block, dtype=np.int32)
    slots = np.empty((batch_size, query_len), dtype=np.int64)
    for i, row in enumerate(rows):
        blocks[i, :widths[i]] = resolved_cpu_pages(
            block_table, row, np.arange(first[i], last[i] + 1, dtype=np.int64), pad_block)
        pages = resolved_cpu_pages(block_table, row, positions[i] // block_size, pad_block)
        slots[i] = np.where(pages == pad_block, pad_slot, pages * block_size + positions[i] % block_size)

    past_positions = first[:, None] * block_size + np.arange(max_blocks * block_size, dtype=np.int64)[None, :]
    past_valid = np.broadcast_to(past_positions[:, None, :] < starts[:, None, None],
                                 (batch_size, query_len, max_blocks * block_size)).copy()
    query_valid = np.ones((batch_size, query_len, query_len), dtype=np.bool_)
    if window is not None:
        past_valid &= past_positions[:, None, :] >= positions[:, :, None] - (window - 1)
        query_valid &= np.abs(positions[:, :, None] - positions[:, None, :]) <= window - 1
    if causal:
        query_valid &= positions[:, None, :] <= positions[:, :, None]
    valid = np.concatenate((past_valid, query_valid), axis=-1)[:, None, :, :]
    bias = np.zeros(valid.shape, dtype=np.float32)
    bias[~valid] = -np.inf
    ids = np.full((batch_size, query_len), mask_token_id, dtype=np.int32)
    ids[:real_batch, 0] = bonus_ids
    return {
        'input_ids': torch.from_numpy(ids),
        'positions': torch.from_numpy(positions),
        'slot_mapping': torch.from_numpy(slots),
        'attn_bias': torch.from_numpy(bias).to(dtype=dtype),
        'block_list': torch.from_numpy(blocks.reshape(-1)) if max_blocks else None,
        'context_lens_tensor': torch.from_numpy((starts - first * block_size).astype(np.int32)),
        'seq_lens_tensor': torch.from_numpy(np.full(batch_size, query_len, dtype=np.int32)),
    }
