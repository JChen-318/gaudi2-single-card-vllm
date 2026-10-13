# SPDX-License-Identifier: Apache-2.0
"""Per-round packed submissions, with no cross-request KV or state cache."""

import torch
from vllm_gaudi.utils import async_h2d_copy


def pack_cpu_tensors(tensors, device):
    """Copy once per dtype, then return same-shape device views in input order.

    Preserve integer address precision and floating usage dtype. Do not cast
    addresses to BF16 or reinterpret bytes using HPU-unsupported dtype views.
    Each invocation owns new storage: previous asynchronous rounds cannot be
    overwritten by host buffer reuse.
    """
    groups = {}
    for index, tensor in enumerate(tensors):
        if tensor.device.type != 'cpu':
            raise ValueError('Packed MTP submission expects CPU metadata')
        groups.setdefault(tensor.dtype, []).append((index, tensor))
    result = [None] * len(tensors)
    for entries in groups.values():
        packed = torch.cat([tensor.reshape(-1) for _, tensor in entries])
        device_packed = async_h2d_copy(packed, device=device)
        offset = 0
        for index, tensor in entries:
            result[index] = device_packed.narrow(0, offset, tensor.numel()).view(tensor.shape)
            offset += tensor.numel()
    return result


def shift_decode_tokens(sampled_tokens, lengths, padded_batch, width):
    """Tensor-only accepted-prefix alignment; lengths are already validated."""
    batch = lengths.shape[0]
    copied_width = min(width, sampled_tokens.shape[1])
    pos = torch.arange(copied_width, device=sampled_tokens.device)
    live = torch.where(pos[None, :] < lengths[:, None], sampled_tokens[:batch, :copied_width], 0).to(torch.int32)
    if copied_width < width:
        live = torch.cat((live, torch.zeros((batch, width - copied_width), dtype=live.dtype, device=live.device)), 1)
    shifted = torch.cat((live, torch.zeros((padded_batch - batch, width), dtype=live.dtype, device=live.device)), 0)
    last_indices = torch.arange(batch, device=lengths.device, dtype=torch.int64) * width + lengths - 1
    return shifted, last_indices


_shift_region = shift_decode_tokens


def configure_submit_region(compile_fn):
    global _shift_region
    _shift_region = compile_fn(shift_decode_tokens)


def prepare_shifted_tokens(sampled_tokens, lengths, padded_batch, width):
    if (sampled_tokens is None or sampled_tokens.ndim != 2 or sampled_tokens.shape[0] < len(lengths)
            or padded_batch < len(lengths)):
        raise ValueError('MTP requires a per-request target sampler output')
    for row, length in enumerate(lengths):
        if not 1 <= length <= min(width, sampled_tokens.shape[1]):
            raise ValueError(f'Accepted MTP prefix length is outside the target verification block: row={row}')
    lengths_device = async_h2d_copy(torch.tensor(lengths, dtype=torch.int64), device=sampled_tokens.device)
    return _shift_region(sampled_tokens, lengths_device, padded_batch, width)
