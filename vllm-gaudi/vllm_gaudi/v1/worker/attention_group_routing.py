# SPDX-License-Identifier: Apache-2.0
"""CPU addressing for legacy attention groups sharing a physical KV pool.

Block tables must already be expanded to attention-kernel pages. Logical
token positions are independent of physical page IDs and M-RoPE axes.
"""
from __future__ import annotations

import torch


def _live_page(table: list[int], logical_page: int, pad_block: int) -> int:
    if logical_page < 0 or logical_page >= len(table):
        raise IndexError("Attention group block table does not cover scheduled tokens")
    page = int(table[logical_page])
    if page < 0 or page >= pad_block:
        raise IndexError("A live attention token cannot address a padding or out-of-range page")
    return page


def prefill_group_addresses(tables: list[list[int]], context_lens: list[int], query_lens: list[int],
                            padded_batch: int, padded_query: int, padded_context_pages: int,
                            block_size: int, pad_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use group-owned pages for chunked-prefill queries and context fetches."""
    if not len(tables) == len(context_lens) == len(query_lens) <= padded_batch:
        raise ValueError("Prefill group addressing requires one table and lengths per real request")
    slots = torch.full((padded_batch, padded_query), -1, dtype=torch.int64)
    context = torch.full((padded_batch, padded_context_pages), -1, dtype=torch.int32)
    for row, (table, start, length) in enumerate(zip(tables, context_lens, query_lens)):
        if start < 0 or not 0 <= length <= padded_query:
            raise ValueError("Invalid prefill query length or context")
        context_count = (start + block_size - 1) // block_size
        if context_count > padded_context_pages:
            raise ValueError("Prefill context page bucket is too small")
        for logical_page in range(context_count):
            context[row, logical_page] = _live_page(table, logical_page, pad_block)
        for token in range(length):
            position = start + token
            page = _live_page(table, position // block_size, pad_block)
            slots[row, token] = page * block_size + position % block_size
    return slots, context.flatten()


def decode_group_addresses(tables: list[list[int]], context_lens: list[int], query_lens: list[int],
                           padded_batch: int, query_width: int, block_size: int,
                           pad_block: int) -> tuple[torch.Tensor, list[list[int]]]:
    """Each flattened verification query sees its own causal page prefix."""
    if not len(tables) == len(context_lens) == len(query_lens) <= padded_batch:
        raise ValueError("Decode group addressing requires one table and lengths per real request")
    slots = torch.empty((padded_batch, query_width), dtype=torch.int32)
    prefixes = []
    dummy_token = 0
    for row in range(padded_batch):
        length = query_lens[row] if row < len(query_lens) else 0
        if not 0 <= length <= query_width:
            raise ValueError("Decode query length exceeds the verification width")
        start = context_lens[row] if row < len(context_lens) else 0
        if start < 0:
            raise ValueError("Decode context cannot be negative")
        table = tables[row] if row < len(tables) else []
        for token in range(query_width):
            if token < length:
                position = start + token
                logical_page = position // block_size
                page = _live_page(table, logical_page, pad_block)
                slots[row, token] = page * block_size + position % block_size
                prefixes.append([_live_page(table, index, pad_block) for index in range(logical_page + 1)])
            else:
                slots[row, token] = pad_block * block_size + dummy_token % block_size
                dummy_token += 1
                prefixes.append([pad_block])
    return slots.reshape(-1, 1), prefixes
