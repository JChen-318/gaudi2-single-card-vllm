# SPDX-License-Identifier: Apache-2.0
"""CPU contracts and tensor layout helpers for causal HPU MTP drafting."""

import torch


def normalize_mtp_positions(positions, num_tokens, uses_mrope, mrope_dims=3):
    """Preserve spatial axes; flatten only the token dimension."""
    if uses_mrope:
        if positions.ndim == 2 and positions.shape == (mrope_dims, num_tokens):
            return positions
        if positions.numel() != num_tokens:
            raise ValueError("MTP positions do not match the draft token layout")
        return positions.reshape(1, num_tokens).expand(mrope_dims, -1).contiguous()
    if positions.ndim == 2 and positions.shape == (mrope_dims, num_tokens):
        positions = positions[0]
    if positions.numel() != num_tokens:
        raise ValueError("MTP positions do not match the draft token layout")
    return positions.reshape(-1)


def select_mtp_positions(positions, indices, uses_mrope):
    return positions.index_select(-1 if uses_mrope else 0, indices.long())


def resolve_mtp_tables(block_table, batch_rows, context_lens, query_lens, block_size, pad_block, resolve_block):
    """Resolve only live pages; never truncate an undersized table silently."""
    if block_table.device.type != "cpu" or block_table.ndim != 2 or block_size <= 0:
        raise ValueError("MTP addressing requires a CPU kernel-page block table")
    if not len(batch_rows) == len(context_lens) == len(query_lens):
        raise ValueError("MTP addressing requires lengths for every real request")
    tables = []
    for row, start, length in zip(batch_rows, context_lens, query_lens, strict=True):
        if not 0 <= row < block_table.shape[0] or start < 0 or length < 0:
            raise ValueError("Invalid MTP request row or sequence length")
        count = (start + length + block_size - 1) // block_size
        if count > block_table.shape[1]:
            raise IndexError(f"MTP KV table has {block_table.shape[1]} columns but request row {row} needs {count}")
        pages = [int(resolve_block(int(page))) for page in block_table[row, :count].tolist()]
        if any(page < 0 or page >= pad_block for page in pages):
            raise IndexError("A live MTP token cannot address a padding or out-of-range KV page")
        tables.append(pages)
    return tables


def mtp_shift_prefill_tokens(token_ids, query_lens, last_input_ids):
    """Shift within each request, including the known next prompt token on a chunk boundary."""
    if token_ids.ndim != 2 or len(query_lens) > token_ids.shape[0] or len(last_input_ids) != len(query_lens):
        raise ValueError("Invalid MTP prefill token layout")
    shifted = torch.zeros_like(token_ids)
    for row, length in enumerate(query_lens):
        if not 0 < length <= token_ids.shape[1]:
            raise ValueError("A real MTP prefill request must contain at least one token")
        if length > 1:
            shifted[row, :length - 1] = token_ids[row, 1:length]
        shifted[row, length - 1] = last_input_ids[row]
    return shifted


def mtp_shift_decode_tokens(sampled_token_ids, accepted_lengths, padded_batch, query_width):
    """Align accepted target outputs with target hidden states, without negative-index gathers."""
    if (sampled_token_ids is None or sampled_token_ids.ndim != 2 or sampled_token_ids.shape[0] < len(accepted_lengths)
            or padded_batch < len(accepted_lengths)):
        raise ValueError("MTP requires a per-request target sampler output")
    shifted = torch.zeros((padded_batch, query_width), dtype=torch.int32, device=sampled_token_ids.device)
    last_indices = []
    for row, length in enumerate(accepted_lengths):
        if not 1 <= length <= min(query_width, sampled_token_ids.shape[1]):
            raise ValueError(f"Accepted MTP prefix length is outside the target verification block: "
                             f"row={row}, accepted={length}, query_width={query_width}, "
                             f"sampler_shape={tuple(sampled_token_ids.shape)}")
        shifted[row, :length] = sampled_token_ids[row, :length].to(torch.int32)
        last_indices.append(row * query_width + length - 1)
    return shifted, torch.tensor(last_indices, dtype=torch.int64, device=sampled_token_ids.device)


def mtp_warmup_target_output(query_lens, padded_batch, query_width):
    """Seed graph capture on CPU; compile-only sampler buffers have no valid values.

    This is a warmup-only scheduling fixture, never an inference sampler.
    Keep host outputs separate from the HPU copy so compile-only execution
    cannot turn a subsequent device-to-host read into uninitialized tokens.
    """
    if len(query_lens) > padded_batch or query_width < 1:
        raise ValueError("Invalid MTP warmup target layout")
    tokens = torch.full((padded_batch, query_width), -1, dtype=torch.int32, device="cpu")
    outputs = []
    for row, length in enumerate(query_lens):
        if not 1 <= length <= query_width:
            raise ValueError("Invalid MTP warmup accepted length")
        tokens[row, :length] = 0
        outputs.append([0] * length)
    return outputs, tokens
