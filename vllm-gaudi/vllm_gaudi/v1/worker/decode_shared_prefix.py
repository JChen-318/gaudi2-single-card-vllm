# SPDX-License-Identifier: Apache-2.0
"""CPU metadata for shared-prefix decode attention.

A DFlash2/MTP verification step schedules up to ``query_width`` tokens per
request. The flattened decode layout gives every token its own causal page
prefix, so a request's KV pages are fetched once per token. Here a request
contributes its pages once and ``block_usage[page, token]`` holds how many
slots of that page the token may attend to (0: the page lies entirely after
the token). Functions return ``None`` for inputs outside their contract.
Nothing here touches the HPU.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch


def _lengths(context_lens: Sequence[int], query_lens: Sequence[int],
             query_width: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    num_real = len(context_lens)
    if num_real == 0 or len(query_lens) < num_real or query_width < 1:
        return None
    try:
        starts = np.asarray([int(v) for v in context_lens], dtype=np.int64)
        lengths = np.asarray([int(v) for v in query_lens[:num_real]], dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if int(starts.min()) < 0 or int(lengths.min()) < 1 or int(lengths.max()) > query_width:
        return None
    return starts, lengths


def shared_prefix_buffers(
    context_lens: Sequence[int],
    query_lens: Sequence[int],
    query_width: int,
    block_size: int,
    table: np.ndarray,
    pad_block: int,
    bucket_blocks: Callable[[int], int],
    usage_dtype: torch.dtype,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Paged-attention buffers with one page list per request.

    ``context_lens[i]`` tokens of request ``i`` are already cached and
    ``query_lens[i]`` (1..query_width) are scheduled; ``table[i]`` holds its
    resolved physical pages. Returns ``(block_list[P], block_groups[P],
    block_usage[P, query_width])``: ``block_groups`` is the request row (-1 for
    padding) and padding tokens inside a row repeat the row's last live token.
    """
    parsed = _lengths(context_lens, query_lens, query_width)
    if parsed is None or table.ndim != 2 or table.shape[0] < len(context_lens):
        return None
    starts, lengths = parsed
    pages = (starts + lengths + block_size - 1) // block_size
    if int(pages.max()) > table.shape[1]:
        return None
    total = int(pages.sum())
    owner = np.repeat(np.arange(pages.size, dtype=np.int64), pages)
    logical = np.arange(total, dtype=np.int64) - (np.cumsum(pages) - pages)[owner]
    entries = table[owner, logical].astype(np.int64, copy=False)
    if int(entries.min()) < 0 or int(entries.max()) >= pad_block:
        return None
    target = max(int(bucket_blocks(total)), total)
    block_list = np.full(target, pad_block, dtype=np.int64)
    block_list[:total] = entries
    block_groups = np.full(target, -1, dtype=np.int64)
    block_groups[:total] = owner
    tokens = np.minimum(np.arange(query_width, dtype=np.int64)[None, :], (lengths[owner] - 1)[:, None])
    visible = starts[owner][:, None] + tokens + 1 - logical[:, None] * block_size
    usage = np.ones((target, query_width), dtype=np.int64)
    usage[:total] = np.clip(visible, 0, block_size)
    return torch.from_numpy(block_list), torch.from_numpy(block_groups), torch.from_numpy(usage).to(usage_dtype)


def group_slots(
    context_lens: Sequence[int],
    query_lens: Sequence[int],
    padded_batch: int,
    query_width: int,
    block_size: int,
    table: np.ndarray,
    pad_block: int,
) -> Optional[torch.Tensor]:
    """KV slots of every scheduled token, as ``decode_group_addresses`` computes them.

    Returns ``slots[padded_batch * query_width, 1]`` (int32); padding tokens
    cycle through the padding page.
    """
    parsed = _lengths(context_lens, query_lens, query_width)
    num_real = len(context_lens)
    if parsed is None or num_real > padded_batch or table.ndim != 2 or table.shape[0] < num_real:
        return None
    starts = np.zeros(padded_batch, dtype=np.int64)
    lengths = np.zeros(padded_batch, dtype=np.int64)
    starts[:num_real], lengths[:num_real] = parsed
    tokens = np.tile(np.arange(query_width, dtype=np.int64), padded_batch)
    rows = np.repeat(np.arange(padded_batch, dtype=np.int64), query_width)
    live = tokens < lengths[rows]
    positions = (starts[rows] + tokens)[live]
    logical = positions // block_size
    if int(logical.max()) >= table.shape[1]:
        return None
    pages = table[rows[live], logical].astype(np.int64, copy=False)
    if int(pages.min()) < 0 or int(pages.max()) >= pad_block:
        return None
    slots = np.empty(padded_batch * query_width, dtype=np.int64)
    slots[live] = pages * block_size + positions % block_size
    dead = np.flatnonzero(~live)
    slots[dead] = pad_block * block_size + np.arange(dead.size, dtype=np.int64) % block_size
    if int(slots.max()) > np.iinfo(np.int32).max:
        return None
    return torch.from_numpy(slots.astype(np.int32)).reshape(-1, 1)
