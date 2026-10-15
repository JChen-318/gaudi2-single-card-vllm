# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.sample import rejection_sampler
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.v1.spec_decode.mtp_phase_profile import profile_stage

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE


def _dflash2_greedy_fastpath_supported(sampling_metadata: SamplingMetadata, ) -> bool:
    """Return whether argmax can bypass the general sampler losslessly."""
    if not sampling_metadata.all_greedy:
        return False
    if sampling_metadata.max_num_logprobs is not None or sampling_metadata.logprob_token_ids:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None or sampling_metadata.bad_words_token_ids:
        return False

    # Spec decode currently installs only MinTokens as an argmax-changing
    # processor. It is a true no-op once its active request map is empty.
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        if not isinstance(processor, MinTokensLogitsProcessor) or processor.min_toks:
            return False

    holder = sampling_metadata.thinking_budget_state_holder
    return holder is None or not holder.has_tracked_requests()


def dflash2_greedy_rejection_sample(
    logits: torch.Tensor,
    metadata: SpecDecodeMetadata,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor | None:
    """Fuse DFlash2's two greedy sampler passes into one vocabulary argmax.

    The general rejection sampler separately gathers/casts/samples bonus and
    target rows. For the common unconstrained greedy request, casting cannot
    change the argmax, so one argmax over the already-packed verification
    logits is exactly equivalent and avoids materializing FP32 vocabulary
    tensors twice.
    """
    if not _dflash2_greedy_fastpath_supported(sampling_metadata):
        return None

    return dflash2_greedy_rejection_sample_packed(
        logits,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.cu_num_draft_tokens,
    )


def dflash2_greedy_rejection_sample_packed(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Tensor-only DFlash2 greedy sampler suitable for regional compile."""
    all_token_ids = logits.argmax(dim=-1).to(torch.int32)
    target_token_ids = torch.index_select(all_token_ids, 0, target_logits_indices)
    bonus_token_ids = torch.index_select(all_token_ids, 0, bonus_logits_indices).view(-1, 1)
    return _rejection_sample_pytorch(
        draft_token_ids,
        target_token_ids,
        bonus_token_ids,
        bonus_logits_indices.shape[0],
        cu_num_draft_tokens,
    )


def rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Performs vectorized rejection sampling on a batch of token sequences.

    This function compares draft tokens to target tokens and accepts them up to 
    the first mismatch. If an entire sequence of draft tokens is accepted, a 
    bonus token is appended. This version handles variable numbers of draft 
    tokens per sequence.

    The current HPU implementation of spec decode will flatten the num_draft_tokens
    to 1. And so the shape size of padded_draft_token_ids will be
    the [real batch size * num_draft_tokens, 1].

    Args:
        padded_draft_token_ids (torch.Tensor): A 2D tensor of draft tokens.
            Shape: (num_seqs * max_draft_tokens, 1)
        padded_target_token_ids (torch.Tensor): A 1D tensor of target tokens
            predicted by the main model.
            Shape: (num_seqs * max_draft_tokens)
        bonus_token_ids (torch.Tensor): A single bonus token for each sequence,
            to be used if all draft tokens are accepted.
            Shape: (num_seqs, 1)
        num_draft_tokens: list[int]: List of number draft tokens for each sequence.
            Shape: (num_seqs)
        cu_num_draft_tokens (torch.Tensor): The cumulative sum of the number of
            draft tokens for each request. Used to determine actual sequence 
            lengths. Shape: (num_seqs,)

    Returns:
        torch.Tensor: The resulting tensor of accepted tokens.
            Shape: (num_seqs, max_draft_tokens + 1)
    """
    return _rejection_sample_pytorch(
        padded_draft_token_ids,
        padded_target_token_ids,
        bonus_token_ids,
        len(num_draft_tokens),
        cu_num_draft_tokens,
    )


def _rejection_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    padded_target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_seqs: int,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    # Keep the prefix comparison and output packing on the input device.  The
    # caller already performs one final D2H copy when parsing engine output;
    # copying all draft/target rows here adds an extra synchronization to every
    # speculative step and is especially expensive for short decode batches.
    padded_draft_token_ids = padded_draft_token_ids.to(torch.int32)
    padded_target_token_ids = padded_target_token_ids.to(torch.int32)
    bonus_token_ids = bonus_token_ids.to(torch.int32)
    padded_draft_token_ids = padded_draft_token_ids.view(num_seqs, -1)
    max_draft_tokens = padded_draft_token_ids.shape[-1]
    padded_target_token_ids = padded_target_token_ids.view(num_seqs, -1)
    bonus_token_ids = bonus_token_ids.view(num_seqs, -1)
    device = padded_draft_token_ids.device

    # Calculate the number of draft tokens for each request without moving the
    # cumulative lengths to the host.
    cu_num_draft_tokens = cu_num_draft_tokens.to(device=device)
    start_indices = torch.cat((torch.zeros(1, device=device,
                                           dtype=cu_num_draft_tokens.dtype), cu_num_draft_tokens[:-1]))
    num_draft_tokens_per_seq = cu_num_draft_tokens - start_indices

    # Find the first mismatch while ignoring padded draft positions.
    pos = torch.arange(max_draft_tokens, device=device, dtype=num_draft_tokens_per_seq.dtype)
    valid_token_mask = pos < num_draft_tokens_per_seq.unsqueeze(-1)
    mismatches = (padded_draft_token_ids != padded_target_token_ids) & valid_token_mask
    any_mismatch = mismatches.any(dim=1)
    first_mismatch_idx = torch.argmax(mismatches.int(), dim=1)

    prefix_len = torch.where(any_mismatch, first_mismatch_idx + 1, num_draft_tokens_per_seq)
    prefix_mask = (pos < prefix_len.unsqueeze(-1)) & valid_token_mask
    placeholder = torch.full_like(padded_target_token_ids, PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    output_prefix = torch.where(prefix_mask, padded_target_token_ids, placeholder)

    output_tokens = torch.cat(
        (output_prefix, placeholder[:, :1]),
        dim=1,
    )
    output_pos = torch.arange(max_draft_tokens + 1, device=device, dtype=num_draft_tokens_per_seq.dtype)
    bonus_mask = (~any_mismatch).unsqueeze(-1) & (output_pos.unsqueeze(0) == num_draft_tokens_per_seq.unsqueeze(-1))
    return torch.where(bonus_mask, bonus_token_ids[:, :1], output_tokens)


def _num_draft_tokens_from_cumsum(cu_num_draft_tokens: torch.Tensor, ) -> torch.Tensor:
    """Convert cumulative draft lengths to per-request lengths on-device."""
    return cu_num_draft_tokens - torch.cat((torch.zeros_like(cu_num_draft_tokens[:1]), cu_num_draft_tokens[:-1]))


def _expand_request_values_for_hpu(
    values: torch.Tensor,
    num_rows: int,
    *,
    replace_greedy_temperature: bool = False,
) -> torch.Tensor:
    """Expand request metadata over HPU's rectangular spec-decode rows.

    Upstream vLLM packs only valid draft rows.  The HPU runner instead keeps a
    fixed-width rectangle and represents unused target rows with index ``-1``.
    Expanding from cumulative valid lengths would therefore shift temperature,
    top-k, and top-p values into the wrong request whenever lengths differ.
    """
    batch_size = values.shape[0]
    if batch_size == 0 or num_rows % batch_size != 0:
        raise ValueError("HPU speculative rows must form a batch-aligned rectangle: "
                         f"rows={num_rows}, batch={batch_size}")
    if replace_greedy_temperature:
        # 1Cat HPU input batches use -1 for greedy, upstream uses 0.
        values = torch.where(values <= GREEDY_TEMPERATURE, torch.ones_like(values), values)
    width = num_rows // batch_size
    return values.unsqueeze(1).expand(batch_size, width).reshape(num_rows)


@profile_stage('sampling_constraints')
def apply_sampling_constraints_hpu(
    logits: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """HPU-safe temperature/top-k/top-p for rectangular verification logits.

    This mirrors upstream ``apply_sampling_constraints`` but deliberately uses
    PyTorch operators.  The upstream expansion and random rejection paths are
    Triton kernels and assume packed CUDA-style rows, neither of which matches
    the HPU runner.
    """
    del cu_num_draft_tokens  # Row ownership is encoded by the HPU rectangle.
    if sampling_metadata.all_greedy:
        return logits
    if sampling_metadata.temperature is None:
        raise ValueError("temperature metadata is required for random sampling")

    num_rows = logits.shape[0]
    if gaudi_envs.VLLM_HPU_SPEC_OPTIMIZATION:
        # Validate the rectangle outside the tensor-only compiled region.
        _expand_request_values_for_hpu(sampling_metadata.temperature, num_rows)
        from vllm_gaudi.v1.spec_decode.spec_tensor_regions import apply_constraints
        return apply_constraints(logits, sampling_metadata.temperature, sampling_metadata.top_k,
                                 sampling_metadata.top_p)
    temperature = _expand_request_values_for_hpu(
        sampling_metadata.temperature,
        num_rows,
        replace_greedy_temperature=True,
    )
    logits.div_(temperature.unsqueeze(-1))

    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = _expand_request_values_for_hpu(sampling_metadata.top_k, num_rows)
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = _expand_request_values_for_hpu(sampling_metadata.top_p, num_rows)
    return apply_top_k_top_p_pytorch(logits, top_k, top_p)


@profile_stage('rejection_uniform_rng')
def _generate_rectangular_uniform_probs(
    batch_size: int,
    width: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Generate acceptance uniforms while preserving per-request RNG state."""
    from vllm_gaudi.v1.worker.kiro_v11 import pooled_draw
    pooled = pooled_draw("uniform", (batch_size, width), torch.float32, device, generators)
    if pooled is not None:  # VLLM_HPU_RNG_POOL
        return pooled
    uniform_probs = torch.rand((batch_size, width), dtype=torch.float32, device=device)
    for req_idx, generator in generators.items():
        if num_draft_tokens[req_idx] > 0:
            uniform_probs[req_idx, :num_draft_tokens[req_idx]].uniform_(generator=generator)
    return uniform_probs


@profile_stage('rejection_exponential_rng')
def _generate_request_exponential_noise(
    batch_size: int,
    vocab_size: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
    use_fp64_gumbel: bool,
) -> torch.Tensor:
    """Generate the exponential-race noise used for recovered-token sampling."""
    dtype = torch.float64 if use_fp64_gumbel else torch.float32
    from vllm_gaudi.v1.worker.kiro_v11 import pooled_draw
    pooled = pooled_draw("exponential", (batch_size, vocab_size), dtype, device, generators)
    if pooled is not None:  # VLLM_HPU_RNG_POOL
        return pooled
    noise = torch.empty((batch_size, vocab_size), dtype=dtype, device=device)
    noise.exponential_()
    for req_idx, generator in generators.items():
        if num_draft_tokens[req_idx] > 0:
            noise[req_idx].exponential_(generator=generator)
    return noise


@profile_stage('sampling_correction')
def rejection_random_sample_pytorch(
    padded_draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    padded_target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Standard rejection correction implemented with HPU-supported PyTorch.

    DFlash2 proposes one deterministic (greedy) block, so ``draft_probs`` is
    normally ``None`` and its proposal distribution is a delta distribution.
    A proposed token is then accepted with probability ``p_target(token)``;
    after a rejection, recovery samples from the target distribution with that
    proposed token removed.  This preserves the target sampler distribution
    while allowing DFlash2 itself to remain deterministic.
    """
    batch_size = len(num_draft_tokens)
    if batch_size == 0:
        return torch.empty((0, 1), dtype=torch.int32, device=padded_target_logits.device)
    num_rows, vocab_size = padded_target_logits.shape
    if num_rows % batch_size != 0:
        raise ValueError("HPU speculative rows must form a batch-aligned rectangle: "
                         f"rows={num_rows}, batch={batch_size}")
    width = num_rows // batch_size
    device = padded_target_logits.device
    if width == 0:
        return bonus_token_ids.to(torch.int32).reshape(batch_size, 1)

    if gaudi_envs.VLLM_HPU_SPEC_OPTIMIZATION:
        if sampling_metadata.temperature is None:
            raise ValueError("temperature metadata is required for random sampling")
        if synthetic_mode and synthetic_conditional_rates is None:
            raise ValueError("synthetic rates are required in synthetic mode")
        # Retain the reference RNG draw shapes, order and per-request overrides.
        uniforms = _generate_rectangular_uniform_probs(batch_size, width, num_draft_tokens,
                                                       sampling_metadata.generators, device)
        noise = _generate_request_exponential_noise(batch_size, vocab_size, num_draft_tokens,
                                                    sampling_metadata.generators, device, use_fp64_gumbel)
        if (device.type == 'hpu' and gaudi_envs.VLLM_HPU_NATIVE_CORRECTION
                and sampling_metadata.all_random and draft_probs is None
                and not synthetic_mode and not use_fp64_gumbel
                and batch_size <= 16 and 1 <= width <= 8 and 64 <= vocab_size <= 1048576
                and all(0 <= length <= width for length in num_draft_tokens)):
            from vllm_gaudi.v1.sample.native_correction import apply_native_correct
            return apply_native_correct(padded_draft_token_ids.reshape(batch_size, width),
                                        cu_num_draft_tokens.to(device=device), padded_target_logits,
                                        bonus_token_ids, uniforms, noise)
        from vllm_gaudi.v1.spec_decode.spec_tensor_regions import apply_correction
        return apply_correction(padded_draft_token_ids.reshape(batch_size, width),
                                cu_num_draft_tokens.to(device=device), draft_probs, padded_target_logits,
                                bonus_token_ids, sampling_metadata.temperature, uniforms, noise,
                                sampling_metadata.all_random, synthetic_conditional_rates if synthetic_mode else None)

    draft_token_ids = padded_draft_token_ids.reshape(batch_size, width).to(torch.int64)
    target_probs = padded_target_logits.softmax(dim=-1, dtype=torch.float32).view(batch_size, width, vocab_size)
    lengths = _num_draft_tokens_from_cumsum(cu_num_draft_tokens.to(device=device)).to(torch.long)
    pos = torch.arange(width, device=device, dtype=lengths.dtype)
    valid_mask = pos.unsqueeze(0) < lengths.unsqueeze(1)
    safe_draft_ids = draft_token_ids.clamp(min=0, max=vocab_size - 1)

    if sampling_metadata.temperature is None:
        raise ValueError("temperature metadata is required for random sampling")
    is_greedy = sampling_metadata.temperature <= GREEDY_TEMPERATURE
    target_argmax = target_probs.argmax(dim=-1)
    greedy_rejected = (draft_token_ids != target_argmax) & valid_mask

    uniform_probs = _generate_rectangular_uniform_probs(
        batch_size,
        width,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )
    target_draft_probs = target_probs.gather(-1, safe_draft_ids.unsqueeze(-1)).squeeze(-1)
    if draft_probs is None:
        proposal_draft_probs = torch.ones_like(target_draft_probs)
    else:
        proposal_probs = draft_probs.reshape(batch_size, width, vocab_size).to(torch.float32)
        proposal_draft_probs = proposal_probs.gather(-1, safe_draft_ids.unsqueeze(-1)).squeeze(-1)

    if synthetic_mode:
        if synthetic_conditional_rates is None:
            raise ValueError("synthetic rates are required in synthetic mode")
        random_accepted = uniform_probs < synthetic_conditional_rates[:width].unsqueeze(0)
    else:
        safe_ratio = torch.where(
            proposal_draft_probs > 0,
            target_draft_probs / proposal_draft_probs,
            torch.zeros_like(target_draft_probs),
        )
        random_accepted = uniform_probs < safe_ratio
    valid_draft_ids = (draft_token_ids >= 0) & (draft_token_ids < vocab_size)
    random_accepted &= valid_mask & valid_draft_ids
    random_rejected = (~random_accepted) & valid_mask

    # Upstream draws one exponential-noise vocabulary vector per request and
    # reuses it for every recovery position in that request.  Preserve that RNG
    # contract here while avoiding Triton-only kernels.
    noise = _generate_request_exponential_noise(
        batch_size,
        vocab_size,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
        use_fp64_gumbel,
    )
    if draft_probs is None:
        recovery_weights = target_probs.clone()
        recovery_weights.scatter_(-1, safe_draft_ids.unsqueeze(-1), 0.0)
        recovery_weights = torch.where(
            valid_draft_ids.unsqueeze(-1),
            recovery_weights,
            target_probs,
        )
    else:
        recovery_weights = torch.clamp(target_probs - proposal_probs, min=0.0)
    recovered_token_ids = (recovery_weights / noise.unsqueeze(1)).argmax(dim=-1)

    greedy_rows = is_greedy.unsqueeze(1)
    rejected = torch.where(greedy_rows, greedy_rejected, random_rejected)
    chosen_tokens = torch.where(
        greedy_rows,
        target_argmax,
        torch.where(random_accepted, draft_token_ids, recovered_token_ids),
    ).to(torch.int32)

    any_rejected = rejected.any(dim=1)
    first_rejected = rejected.to(torch.int32).argmax(dim=1).to(lengths.dtype)
    output_len = torch.where(any_rejected, first_rejected + 1, lengths)
    output_mask = (pos.unsqueeze(0) < output_len.unsqueeze(1)) & valid_mask
    placeholder = torch.full_like(chosen_tokens, PLACEHOLDER_TOKEN_ID)
    output_prefix = torch.where(output_mask, chosen_tokens, placeholder)
    output_tokens = torch.cat((output_prefix, placeholder[:, :1]), dim=1)

    output_pos = torch.arange(width + 1, device=device, dtype=lengths.dtype)
    bonus_mask = (~any_rejected).unsqueeze(1) & (output_pos.unsqueeze(0) == lengths.unsqueeze(1))
    return torch.where(bonus_mask, bonus_token_ids.to(torch.int32).view(batch_size, 1), output_tokens)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    if sampling_metadata.all_greedy:
        target_argmax = target_logits.argmax(dim=-1)
        return rejection_sample_pytorch(draft_token_ids, target_argmax, bonus_token_ids, num_draft_tokens,
                                        cu_num_draft_tokens)

    return rejection_random_sample_pytorch(
        draft_token_ids,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata,
        synthetic_mode,
        synthetic_conditional_rates,
        use_fp64_gumbel,
    )


rejection_sampler.rejection_sample = rejection_sample
rejection_sampler.apply_sampling_constraints = apply_sampling_constraints_hpu
