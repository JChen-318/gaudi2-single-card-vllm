# SPDX-License-Identifier: Apache-2.0
"""Real Gaudi2 T8-kernel composition versus the independent T16 recurrence."""

import torch
from flashinfer_gaudi import gdn_decode as gdn
from flashinfer_gaudi import load_native_extensions

load_native_extensions()
torch.set_num_threads(4)
gen = torch.Generator().manual_seed(81)


class CachedRegion(torch.nn.Module):
    """Match a model region: mutate an aliased cache and return only activations."""

    def __init__(self, pool, direct):
        super().__init__()
        self.register_buffer("cache", torch.stack((torch.zeros_like(pool), pool)))
        self.direct = direct

    def forward(self, packed, decay, beta, slots, accepted):
        output, _ = gdn._gated_delta_rule_mtp_fused16(
            packed, decay, beta, self.cache[1], slots, accepted,
            assume_distinct_checkpoints=True, checkpoint_start=1 if self.direct else None)
        return output


for batch, accepted_value, direct in ((1, 9, False), (2, 16, False), (1, 9, True), (2, 16, True)):
    pool = torch.randn(40, 48, 128, 128, generator=gen) * 0.05
    packed = torch.randn(batch, 16, 10240, generator=gen).bfloat16()
    decay = -torch.rand(batch, 16, 48, generator=gen) * 0.5
    beta = torch.rand(batch, 16, 48, generator=gen).bfloat16()
    slots = torch.randperm(39, generator=gen)[:batch * 16].add(1).reshape(batch, 16).int()
    if direct:
        slots = torch.arange(1, batch * 16 + 1, dtype=torch.int32).reshape(batch, 16)
    accepted = torch.full((batch,), accepted_value, dtype=torch.int32)
    lengths = torch.full((batch,), 16, dtype=torch.int32)
    ref, ref_pool = gdn._gated_delta_rule_mtp_packed_reference(
        packed, decay, beta, pool.clone(), slots, accepted, lengths, assume_full_query=True)
    hp = [x.to("hpu") for x in (packed, decay, beta, pool, slots, accepted)]
    for compiled in (False, True):
        hp[3].copy_(pool.to("hpu"))
        fn = torch.compile(gdn._gated_delta_rule_mtp_fused16, backend="hpu_backend", fullgraph=True,
                           dynamic=False) if compiled else gdn._gated_delta_rule_mtp_fused16
        actual, actual_pool = fn(*hp, assume_distinct_checkpoints=True, checkpoint_start=1 if direct else None)
        torch.testing.assert_close(actual.cpu(), ref, rtol=0.03, atol=0.015)
        torch.testing.assert_close(actual_pool.cpu(), ref_pool, rtol=5e-4, atol=2e-5)
        print("PASS real T16 GDN", batch, accepted_value, "direct", direct, "compiled", compiled, flush=True)
    region = CachedRegion(pool, direct).to("hpu")
    inputs = (hp[0], hp[1], hp[2], hp[4], hp[5])
    for compiled in (False, True):
        region.cache[1].copy_(pool.to("hpu"))
        fn = (torch.compile(region, backend="hpu_backend", fullgraph=True, dynamic=False) if compiled else region)
        actual = fn(*inputs)
        torch.testing.assert_close(actual.cpu(), ref, rtol=0.03, atol=0.015)
        torch.testing.assert_close(region.cache[1].cpu(), ref_pool, rtol=5e-4, atol=2e-5)
        torch.testing.assert_close(region.cache[0].cpu(), torch.zeros_like(pool), rtol=0, atol=0)
        print("PASS hidden aliased T16 cache", batch, "direct", direct, "compiled", compiled, flush=True)
