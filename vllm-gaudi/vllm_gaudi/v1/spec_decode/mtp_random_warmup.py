# SPDX-License-Identifier: Apache-2.0
"""Warm the real rejection entry without touching model/KV/request state."""
from dataclasses import replace
from types import SimpleNamespace

import torch


def warmup_random_mtp(runner):
    cpu_state = torch.random.get_rng_state()
    hpu_state = torch.hpu.get_rng_state()
    count = 0
    try:
        vocab = runner.input_batch.vocab_size
        width = runner.speculative_config.num_speculative_tokens
        max_batch = min(runner.input_batch.max_num_reqs, 4)
        for batch in sorted({1, max_batch}):
            logits = (torch.arange(vocab, device=runner.device, dtype=torch.float32) / vocab).repeat(
                batch * (width + 1), 1)
            target = torch.tensor([row * (width + 1) + step for row in range(batch)
                                   for step in range(width)], device=runner.device, dtype=torch.long)
            bonus = torch.tensor([row * (width + 1) + width for row in range(batch)],
                                 device=runner.device, dtype=torch.long)
            spec = SimpleNamespace(max_spec_len=width, bonus_logits_indices=bonus, target_logits_indices=target,
                                   draft_token_ids=torch.zeros((batch, width), dtype=torch.int32, device=runner.device),
                                   num_draft_tokens=[width] * batch,
                                   cu_num_draft_tokens=torch.arange(1, batch + 1, device=runner.device,
                                                                   dtype=torch.int32) * width)
            seed_modes = [(), tuple(range(batch))]
            if batch > 1:
                seed_modes.append((0,))
            for seeded in seed_modes:
                generators = {row: torch.Generator(device=runner.device).manual_seed(9123 + row) for row in seeded}
                metadata = replace(
                    runner.input_batch.sampling_metadata,
                    temperature=torch.full((batch,), .7, dtype=torch.float32, device=runner.device),
                    all_greedy=False, all_random=True,
                    top_k=torch.full((batch,), 20, dtype=torch.int32, device=runner.device),
                    top_p=torch.full((batch,), .9, dtype=torch.float32, device=runner.device), generators=generators,
                    max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
                    output_token_ids=[[] for _ in range(batch)], allowed_token_ids_mask=None,
                    bad_words_token_ids={}, logprob_token_ids=None, spec_token_ids=None,
                    thinking_budget_state_holder=None)
                # The same RejectionSampler calls the rebound bonus sampler,
                # constraints and correction, including per-request generators.
                for _ in range(2):
                    runner.rejection_sampler(spec, None, logits.clone(), metadata).sampled_token_ids.cpu()
                count += 1
    finally:
        torch.hpu.synchronize()
        torch.hpu.set_rng_state(hpu_state)
        torch.random.set_rng_state(cpu_state)
    if not torch.equal(torch.hpu.get_rng_state(), hpu_state) or not torch.equal(torch.random.get_rng_state(), cpu_state):
        raise RuntimeError('Random MTP warmup failed to restore global RNG states')
    return count
