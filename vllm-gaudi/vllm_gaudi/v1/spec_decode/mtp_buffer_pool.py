# SPDX-License-Identifier: Apache-2.0
"""Bounded metadata storage, reused only after all consumers finish."""
from functools import wraps

import torch
from vllm_gaudi.v1.spec_decode.mtp_submit import pack_cpu_tensors


class MTPMetadataBufferPool:
    def __init__(self, device, event_factory=None, max_keys=8, max_bytes=8 * 1024 * 1024):
        self.device = torch.device(device)
        self.event_factory = event_factory or torch.hpu.Event
        self.max_keys = max_keys
        self.max_bytes = max_bytes
        self.keys = {}
        self.bytes = 0
        self.active = []
        self.stats = {'created': 0, 'reused': 0, 'busy_fallback': 0, 'capacity_fallback': 0}

    def pack(self, tensors):
        if any(tensor.device.type != 'cpu' for tensor in tensors):
            raise ValueError('MTP metadata pool takes CPU tensors only')
        if self.active:
            raise RuntimeError('Previous MTP buffer lease has not been released')
        key = tuple((tensor.dtype, tuple(tensor.shape)) for tensor in tensors)
        size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if key not in self.keys:
            if len(self.keys) >= self.max_keys or self.bytes + 2 * size > self.max_bytes:
                self.stats['capacity_fallback'] += 1
                return pack_cpu_tensors(tensors, self.device)
            self.keys[key] = []
        slots = self.keys[key]
        slot = next((item for item in slots if not item['leased']
                     and (not item['recorded'] or item['event'].query())), None)
        if slot is None and len(slots) < 2 and self.bytes + size <= self.max_bytes:
            grouped = {}
            for index, tensor in enumerate(tensors):
                grouped.setdefault(tensor.dtype, []).append((index, tensor))
            views = [None] * len(tensors)
            buffers = []
            for dtype, entries in grouped.items():
                count = sum(tensor.numel() for _, tensor in entries)
                # This plugin explicitly reports pin_memory unsupported on HPU.
                # Retain host storage through the final-consumer event instead.
                host = torch.empty(count, dtype=dtype, device='cpu')
                device = torch.empty(count, dtype=dtype, device=self.device)
                offset = 0
                for index, tensor in entries:
                    views[index] = device.narrow(0, offset, tensor.numel()).view(tensor.shape)
                    offset += tensor.numel()
                buffers.append((host, device, [index for index, _ in entries]))
            slot = {'buffers': buffers, 'views': views, 'event': self.event_factory(), 'leased': False, 'recorded': False}
            slots.append(slot)
            self.bytes += size
            self.stats['created'] += 1
        elif slot is not None:
            self.stats['reused'] += 1
        if slot is None:
            self.stats['busy_fallback'] += 1
            return pack_cpu_tensors(tensors, self.device)
        slot['leased'] = True
        self.active.append(slot)
        for host, device, indices in slot['buffers']:
            offset = 0
            for index in indices:
                tensor = tensors[index]
                host.narrow(0, offset, tensor.numel()).copy_(tensor.reshape(-1))
                offset += tensor.numel()
            device.copy_(host, non_blocking=True)
        return slot['views']

    def release(self):
        # Record after stack(drafts), not after H2D: views are consumed throughout
        # the causal draft loop. Inference failures still record submitted work.
        # If record fails, the slot stays leased and is never reused.
        for slot in self.active:
            slot['event'].record()
            slot['recorded'] = True
            slot['leased'] = False
        self.active.clear()


def release_mtp_buffers(function):
    @wraps(function)
    def wrapped(proposer, *args, **kwargs):
        try:
            return function(proposer, *args, **kwargs)
        finally:
            pool = getattr(proposer, '_mtp_metadata_pool', None)
            if pool is not None:
                pool.release()
    return wrapped
