# SPDX-License-Identifier: Apache-2.0
"""Vectorised CPU metadata builders for DFlash2/MTP decode.

Each function returns exactly what the list-based reference code in
``hpu_model_runner`` / ``attention_group_routing`` returns, or ``None`` when an
input is outside the contract it has been checked for; callers then run the
reference path, which also owns error reporting. Nothing here touches the HPU.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np
import torch


def gdn_state_indices(
    num_groups: int,
    table_of: Callable[[int], torch.Tensor],
    compact_group_offset: dict[int, int],
    base_slots: Sequence[int],
    gdn_max_reqs: int,
    slots_per_req: int,
    req_indices: Sequence[int],
    block_table_offsets: Sequence[int],
    target_bs: int,
    state_width: int,
    checkpoint_offsets: Optional[Sequence[int]],
    mamba_pad_block_id: int,
) -> Optional[torch.Tensor]:
    """Equivalent of ``HPUModelRunner.prepare_mamba_state_idxs``.

    ``table_of(g)`` returns the CPU block table of non-compact KV group ``g``;
    ``compact_group_offset`` maps compact GDN group ids to their offset and
    ``base_slots[i]`` is the compact base slot of ``req_indices[i]``.
    """
    num_indices = len(req_indices)
    if num_groups == 0 or num_indices > target_bs or state_width < 1:
        return None
    try:
        req_rows = [int(v) for v in req_indices]
        table_offsets = [int(v) for v in block_table_offsets]
    except (TypeError, ValueError):
        return None
    if len(table_offsets) != num_indices:
        return None
    if checkpoint_offsets is not None and (state_width != 1 or len(checkpoint_offsets) != num_indices):
        return None
    out = np.empty((num_groups, target_bs, state_width), dtype=np.int32)
    if compact_group_offset:
        if len(base_slots) != num_indices:
            return None
        base = np.asarray(base_slots, dtype=np.int64)
        if checkpoint_offsets is None:
            offsets = np.broadcast_to(np.arange(state_width, dtype=np.int64), (num_indices, state_width))
        else:
            clipped = np.clip(np.asarray([int(v) for v in checkpoint_offsets], dtype=np.int64), 0, slots_per_req - 1)
            offsets = clipped.reshape(num_indices, 1)
    for group_idx in range(num_groups):
        g_offset = compact_group_offset.get(group_idx)
        if g_offset is not None:
            first_slot = (g_offset * gdn_max_reqs + base) * slots_per_req + 1
            out[group_idx, :num_indices] = first_slot[:, None] + offsets
            out[group_idx, num_indices:] = -1
        else:
            table = table_of(group_idx)
            if table.dtype != torch.int32:
                return None
            values = table[req_rows, table_offsets].numpy()
            out[group_idx, :num_indices] = values.reshape(num_indices, 1)
            out[group_idx, num_indices:] = mamba_pad_block_id
    result = torch.from_numpy(out)
    return result.squeeze(-1) if state_width == 1 else result


class TokenLayout(NamedTuple):
    """Addressing shared by every attention group for one flattened decode batch.

    Token ``t`` belongs to request ``rows[t]``; a live token sees the causal
    page prefix ``[0, logical_page[t]]`` of its request, a dead (padding) token
    sees one padding page. ``owner``/``offset`` enumerate all visible pages.
    """
    block_size: int
    num_tokens: int
    total: int                 # visible pages over all tokens
    live_tokens: np.ndarray    # token indices that are live
    live_rows: np.ndarray      # request row of each live token
    live_pages: np.ndarray     # logical page holding each live token
    live_offsets: np.ndarray   # position % block_size of each live token
    dead_tokens: np.ndarray    # token indices that are padding
    owner: np.ndarray          # token index of each visible page
    last_page: np.ndarray      # index into the page list of each token's final page
    live_entries: Optional[np.ndarray]   # page-list indices owned by live tokens (None: all)
    entry_rows: np.ndarray     # request row of each live page-list entry
    entry_pages: np.ndarray    # logical page of each live page-list entry
    max_page: int              # largest logical page referenced (-1: no live token)


def token_layout(context_lens: Sequence[int], query_lens: Sequence[int], padded_batch: int, query_width: int,
                 block_size: int) -> Optional[TokenLayout]:
    num_real = len(context_lens)
    if len(query_lens) < num_real or num_real > padded_batch or query_width < 1:
        return None
    lengths = np.zeros(padded_batch, dtype=np.int64)
    starts = np.zeros(padded_batch, dtype=np.int64)
    try:
        lengths[:num_real] = [int(v) for v in query_lens[:num_real]]
        starts[:num_real] = [int(v) for v in context_lens]
    except (TypeError, ValueError):
        return None
    if int(lengths.min()) < 0 or int(lengths.max()) > query_width or int(starts.min()) < 0:
        return None
    num_tokens = padded_batch * query_width
    tokens = np.tile(np.arange(query_width, dtype=np.int64), padded_batch)
    rows = np.repeat(np.arange(padded_batch, dtype=np.int64), query_width)
    live = tokens < lengths[rows]
    positions = starts[rows] + tokens
    logical_page = positions // block_size
    pages = np.where(live, logical_page + 1, 1)
    last_page = np.cumsum(pages) - 1
    total = int(last_page[-1]) + 1
    owner = np.repeat(np.arange(num_tokens, dtype=np.int64), pages)
    live_tokens = np.flatnonzero(live)
    all_live = live_tokens.size == num_tokens
    entry_offset = np.arange(total, dtype=np.int64) - (last_page - pages + 1)[owner]
    if all_live:
        live_entries = None
        entry_rows, entry_pages = rows[owner], entry_offset
    else:
        live_entries = np.flatnonzero(live[owner])
        entry_rows, entry_pages = rows[owner[live_entries]], entry_offset[live_entries]
    return TokenLayout(block_size, num_tokens, total, live_tokens, rows[live_tokens], logical_page[live_tokens],
                       positions[live_tokens] % block_size, np.flatnonzero(~live), owner, last_page, live_entries,
                       entry_rows, entry_pages, int(logical_page[live_tokens].max()) if live_tokens.size else -1)


def causal_prefix_buffers(
    layout: TokenLayout,
    table: np.ndarray,
    slots: np.ndarray,
    pad_block: int,
    bucket_blocks: Callable[[int], int],
    usage_dtype: torch.dtype,
    require_live_pages: bool = False,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Equivalent of non-contiguous ``get_habana_paged_attn_buffers`` for per-token causal page prefixes.

    ``table[row]`` holds the request's resolved physical pages and ``slots[t]``
    the KV slot of token ``t``. With ``require_live_pages`` every page seen by a
    live token must be a real page, as ``decode_group_addresses`` enforces.
    """
    if layout.max_page >= table.shape[1] or slots.size != layout.num_tokens:
        return None
    entries = table[layout.entry_rows, layout.entry_pages].astype(np.int64, copy=False)
    if require_live_pages and entries.size and (int(entries.min()) < 0 or int(entries.max()) >= pad_block):
        return None
    total = layout.total
    target = max(int(bucket_blocks(total)), total)
    block_list = np.full(target, pad_block, dtype=np.int64)
    if layout.live_entries is None:
        block_list[:total] = entries
    else:
        block_list[layout.live_entries] = entries
    block_groups = np.full(target, -1, dtype=np.int64)
    block_groups[:total] = layout.owner
    usage = np.ones(target, dtype=np.int64)
    usage[:total] = layout.block_size
    usage[layout.last_page] = slots % layout.block_size + 1
    return torch.from_numpy(block_list), torch.from_numpy(block_groups), torch.from_numpy(usage).to(usage_dtype)


def decode_group_buffers(
    layout: TokenLayout,
    table: np.ndarray,
    pad_block: int,
    bucket_blocks: Callable[[int], int],
    usage_dtype: torch.dtype,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Equivalent of ``decode_group_addresses`` followed by ``get_habana_paged_attn_buffers``.

    ``table`` is ``[num_real_requests, pages]`` with resolved physical page ids.
    Returns ``(slots[N,1] int32, block_list, block_groups, block_usage)``.
    """
    if table.ndim != 2 or layout.max_page >= table.shape[1]:
        return None
    if layout.live_rows.size and int(layout.live_rows.max()) >= table.shape[0]:
        return None
    block_size = layout.block_size
    slots = np.empty(layout.num_tokens, dtype=np.int64)
    slots[layout.live_tokens] = (table[layout.live_rows, layout.live_pages].astype(np.int64, copy=False) * block_size
                                 + layout.live_offsets)
    slots[layout.dead_tokens] = pad_block * block_size + np.arange(layout.dead_tokens.size, dtype=np.int64) % block_size
    buffers = causal_prefix_buffers(layout, table, slots, pad_block, bucket_blocks, usage_dtype,
                                    require_live_pages=True)
    if buffers is None or int(slots.max()) > np.iinfo(np.int32).max:
        return None
    return (torch.from_numpy(slots.astype(np.int32)).reshape(-1, 1), *buffers)
