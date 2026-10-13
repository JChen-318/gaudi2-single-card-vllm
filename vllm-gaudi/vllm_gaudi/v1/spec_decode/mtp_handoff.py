# SPDX-License-Identifier: Apache-2.0
"""Round-local metadata fusion; no KV, RNG or request state is cached."""
import torch


def batch_decode_masks(usages, groups, batch_size, block_size, dtype):
    """One submission for all causal draft-step masks and page mappings."""
    usage = torch.cat(usages)
    owners = torch.cat(groups)
    offsets = torch.arange(block_size, device=usage.device, dtype=torch.int32)
    mask = offsets[None, :] >= usage[:, None]
    bias = torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, float('-inf'))
    # Equality is one_hot for live owners and zero for padding owners (-1).
    # It also preserves zero mappings for the plugin's batch-size sentinel.
    rows = torch.arange(batch_size, device=owners.device, dtype=owners.dtype)
    mapping = (owners[:, None] == rows[None, :]).to(dtype)
    return bias, mapping


_batch_region = batch_decode_masks


def configure_handoff_region(compile_fn):
    global _batch_region
    _batch_region = compile_fn(batch_decode_masks)


def prepare_metadata_batch(adapter, metadata):
    """Preserve the adapter metadata type and fall back for extra mask modes."""
    from vllm_gaudi.v1.worker.hpu_model_runner import (
        custom_tuple_replace,
        trim_attn_metadata,
    )

    if (not metadata or adapter.pooling_model or adapter.metadata_processor.interleaved_sliding_window
            or not getattr(adapter.model, '_hpu_mtp_flatten_hidden', False)
            or any(item.is_prompt or item.attention_groups is not None
                   or item.window_block_list is not None or item.chunked_block_list is not None
                   for item in metadata)):
        return metadata, False
    batch_size = metadata[0].slot_mapping.shape[0]
    block_size = metadata[0].block_size
    if any(item.slot_mapping.shape != (batch_size, 1) or item.block_size != block_size for item in metadata):
        return metadata, False
    # Keep the fused region bounded even for very long contexts. This cap is
    # the bytes of mask/mapping outputs; larger cases retain the step path.
    pages = sum(item.block_usage.numel() for item in metadata)
    if pages * (block_size + batch_size) * torch.empty((), dtype=adapter.dtype).element_size() > 8 * 1024**2:
        return metadata, False
    bias, mapping = _batch_region(tuple(item.block_usage for item in metadata),
                                  tuple(item.block_groups for item in metadata),
                                  batch_size, block_size, adapter.dtype)
    prepared = []
    offset = 0
    for item in metadata:
        count = item.block_usage.numel()
        prepared.append(custom_tuple_replace(trim_attn_metadata(item), 'TrimmedAttentionMetadata',
                                              attn_bias=bias.narrow(0, offset, count),
                                              block_mapping=mapping.narrow(0, offset, count)))
        offset += count
    return prepared, True


def flatten_sampled_outputs(decode_outputs, prefill_tensors):
    """Keep parsed MTP tokens as Python lists until scheduler handoff."""
    result = [token for row in decode_outputs for token in row]
    for tensor in prefill_tensors:
        result.extend(tensor.cpu().reshape(-1).tolist())
    return result
