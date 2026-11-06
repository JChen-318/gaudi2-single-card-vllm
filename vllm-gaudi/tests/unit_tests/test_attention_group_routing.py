# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.v1.worker.attention_group_routing import prefill_group_addresses


def _reference(tables, context_lens, query_lens, padded_batch, padded_query, padded_context_pages, block_size):
    slots = torch.full((padded_batch, padded_query), -1, dtype=torch.int64)
    context = torch.full((padded_batch, padded_context_pages), -1, dtype=torch.int32)
    for row, (table, start, length) in enumerate(zip(tables, context_lens, query_lens)):
        for page in range((start + block_size - 1) // block_size):
            context[row, page] = table[page]
        for token in range(length):
            position = start + token
            slots[row, token] = table[position // block_size] * block_size + position % block_size
    return slots, context.flatten()


@pytest.mark.parametrize("start,length", [(0, 0), (0, 7), (0, 2048), (16, 32), (2048, 2048), (2051, 1)])
def test_prefill_addresses_match_the_per_token_definition(start, length):
    block_size = 16
    pages = (start + length + block_size - 1) // block_size
    table = [(page * 7 + 3) % 1000 for page in range(pages + 2)]
    got = prefill_group_addresses([table], [start], [length], 2, 2048, 300, block_size, 1000)
    want = _reference([table], [start], [length], 2, 2048, 300, block_size)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])
    assert got[0].dtype == torch.int64 and got[1].dtype == torch.int32


@pytest.mark.parametrize("table", [[1], [1, 1000], [1, -1]])
def test_prefill_addresses_reject_unusable_pages(table):
    with pytest.raises(IndexError):
        prefill_group_addresses([table], [0], [32], 1, 32, 1, 16, 1000)


def test_prefill_addresses_reject_short_context_bucket():
    with pytest.raises(ValueError):
        prefill_group_addresses([[1, 2, 3]], [32], [1], 1, 1, 1, 16, 1000)
