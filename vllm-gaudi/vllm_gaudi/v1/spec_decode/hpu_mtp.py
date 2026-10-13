# SPDX-License-Identifier: Apache-2.0
"""Causal Qwen MTP proposer using HPU attention and group-owned KV addressing.

Only proposal tokens are deterministic. Target sampling and rejection remain
under the request's original sampling parameters.
"""

import copy

import torch
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.utils import async_h2d_copy
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm_gaudi.v1.spec_decode.mtp_buffer_pool import release_mtp_buffers
from vllm_gaudi.v1.spec_decode.mtp_optimization_control import enabled as mtp_opt_enabled
from vllm_gaudi.v1.spec_decode.mtp_phase_profile import phase as mtp_phase
from vllm_gaudi.v1.spec_decode.mtp_phase_profile import profile_stage
from vllm_gaudi.v1.spec_decode.mtp_utils import (
    normalize_mtp_positions,
    resolve_mtp_tables,
    select_mtp_positions,
)
from vllm_gaudi.v1.worker.attention_group_routing import (
    decode_group_addresses,
    prefill_group_addresses,
)


class HpuMTPProposer(EagleProposer):
    """HPU adapter for Qwen3.5's existing upstream prediction head."""

    def load_model(self, target_model):
        super().load_model(target_model)
        # The adapter receives bucketed [B,S] tokens, but the upstream MTP
        # decoder and embeddings must both consume flat [B*S,H] activations.
        self.model._hpu_mtp_flatten_hidden = True
        from vllm_gaudi.models.qwen3_next import configure_hpu_qwen3_gated_fp8

        configure_hpu_qwen3_gated_fp8(self.model)

    def get_hpu_adapter_config(self):
        config = copy.copy(self._create_draft_vllm_config())
        config.model_config = self.draft_model_config
        return config

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        """Resolve HPU pages without constructing CUDA metadata builders."""
        groups = [(gid, group) for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                  if self._draft_attn_layer_names & set(group.layer_names)]
        if len(groups) != 1 or not self._draft_attn_layer_names:
            raise ValueError("HPU MTP requires all prediction-head attention layers in one KV group")
        gid, group = groups[0]
        if not self._draft_attn_layer_names <= set(group.layer_names) or hasattr(group.kv_cache_spec, "mamba_type"):
            raise ValueError("HPU MTP prediction-head KV group must contain its complete attention layer set")
        self.kv_cache_gid = gid
        self.block_size = (int(kernel_block_sizes[gid])
                           if kernel_block_sizes is not None else int(group.kv_cache_spec.block_size))
        if self.block_size <= 0:
            raise ValueError("HPU MTP kernel page size must be positive")

    def _tables(self, runner, batch_rows, starts, lengths):
        if self.kv_cache_gid < 0 or self.block_size <= 0:
            raise RuntimeError("HPU MTP attention backend has not been initialized")
        table = runner.input_batch.block_table[self.kv_cache_gid].get_cpu_tensor()
        return resolve_mtp_tables(table, batch_rows, starts, lengths, self.block_size, runner._PAD_BLOCK_ID,
                                  runner._resolve_block)

    def prefill_metadata(self, runner, batch_rows, starts, lengths, padded_batch, query_width, context_pages):
        tables = self._tables(runner, batch_rows, starts, lengths)
        slots, blocks = prefill_group_addresses(tables, starts, lengths, padded_batch, query_width, context_pages,
                                                self.block_size, runner._PAD_BLOCK_ID)
        return HPUAttentionMetadataV1.make_prefill_metadata(
            attn_bias=None,
            block_list=async_h2d_copy(blocks, device=self.device) if context_pages else None,
            context_lens_tensor=async_h2d_copy(torch.tensor(starts + [0] * (padded_batch - len(starts)),
                                                            dtype=torch.int32),
                                               device=self.device),
            seq_lens_tensor=async_h2d_copy(torch.tensor(lengths + [0] * (padded_batch - len(lengths)),
                                                        dtype=torch.int32),
                                           device=self.device),
            slot_mapping=async_h2d_copy(slots, device=self.device),
            block_size=self.block_size,
        )

    @profile_stage('draft_cpu_metadata_and_h2d')
    def decode_metadata(self, runner, batch_rows, starts, lengths, padded_batch, query_width):
        blocks, owners, usage, slots = self._decode_metadata_cpu(runner, batch_rows, starts, lengths, padded_batch,
                                                                 query_width)
        if gaudi_envs.VLLM_HPU_SPEC_OPTIMIZATION:
            from vllm_gaudi.v1.spec_decode.mtp_submit import pack_cpu_tensors
            tensors = pack_cpu_tensors([blocks, owners, usage, slots], self.device)
        else:
            tensors = [async_h2d_copy(tensor, device=self.device) for tensor in (blocks, owners, usage, slots)]
        return self._decode_metadata_from_tensors(*tensors)

    def _decode_metadata_cpu(self, runner, batch_rows, starts, lengths, padded_batch, query_width, tables=None):
        if tables is None:
            tables = self._tables(runner, batch_rows, starts, lengths)
        slots, prefixes = decode_group_addresses(tables, starts, lengths, padded_batch, query_width, self.block_size,
                                                 runner._PAD_BLOCK_ID)
        blocks, owners, usage = runner.get_habana_paged_attn_buffers(prefixes,
                                                                     slots.tolist(),
                                                                     padded_batch * query_width,
                                                                     block_size=self.block_size)
        return blocks, owners, usage, slots

    def _decode_metadata_from_tensors(self, blocks, owners, usage, slots):
        return HPUAttentionMetadataV1.make_decode_metadata(
            block_list=blocks,
            block_usage=usage,
            block_groups=owners,
            input_positions=None,
            slot_mapping=slots,
            block_size=self.block_size,
            window_block_list=None,
            window_block_usage=None,
            window_block_groups=None,
            chunked_block_list=None,
            chunked_block_usage=None,
            chunked_block_groups=None,
        )

    @profile_stage('draft_cpu_metadata_and_h2d')
    def prepare_decode_steps(self, runner, batch_rows, last_positions_cpu):
        from vllm_gaudi.v1.spec_decode.mtp_submit import pack_cpu_tensors

        # Slots/pages depend only on logical position, not on proposed token IDs.
        # Build all six causal metadata sets once, before recursive head launches.
        positions = [int(position) for position in last_positions_cpu]
        tables = None
        if self.num_speculative_tokens > 1:
            future_starts = [position + 1 if position + 1 < self.max_model_len else 0 for position in positions]
            future_lengths = [
                max(0, min(self.num_speculative_tokens - 1, self.max_model_len - position - 1))
                for position in positions
            ]
            # The scheduler allocated the round's complete draft page range.
            # Resolve/validate it once; every step still gets its causal prefix.
            tables = self._tables(runner, batch_rows, future_starts, future_lengths)
        tensors = []
        for step in range(1, self.num_speculative_tokens):
            advanced = [position + step for position in positions]
            lengths = [int(position < self.max_model_len) for position in advanced]
            starts = [position if length else 0 for position, length in zip(advanced, lengths, strict=True)]
            tensors.extend(self._decode_metadata_cpu(runner, batch_rows, starts, lengths, len(batch_rows), 1, tables))
        alive = torch.tensor([[position + step + 1 < self.max_model_len for position in positions]
                              for step in range(self.num_speculative_tokens)],
                             dtype=torch.int64)
        tensors.append(alive)
        if getattr(gaudi_envs, 'VLLM_HPU_MTP_METADATA_POOL', False):
            from vllm_gaudi.v1.spec_decode.mtp_buffer_pool import MTPMetadataBufferPool
            if not hasattr(self, '_mtp_metadata_pool'):
                self._mtp_metadata_pool = MTPMetadataBufferPool(self.device)
            packed = self._mtp_metadata_pool.pack(tensors)
        else:
            packed = pack_cpu_tensors(tensors, self.device)
        metadata = [
            self._decode_metadata_from_tensors(*packed[offset:offset + 4]) for offset in range(0,
                                                                                               len(packed) - 1, 4)
        ]
        if mtp_opt_enabled('handoff'):
            from vllm_gaudi.v1.spec_decode.mtp_handoff import prepare_metadata_batch
            with mtp_phase('draft_batched_metadata_processing'):
                metadata, _ = prepare_metadata_batch(self.model, metadata)
        return metadata, packed[-1]

    @profile_stage('draft_generation')
    @release_mtp_buffers
    def propose(self, token_ids, positions, target_hidden_states, last_indices, metadata, runner, batch_rows,
                last_positions_cpu):
        """Warm the accepted prefix once, then recursively feed MTP output states."""
        num_tokens = token_ids.numel()
        positions = normalize_mtp_positions(positions, num_tokens, self.uses_mrope,
                                            getattr(self.draft_model_config, "mrope_num_dims", 3))
        hidden = target_hidden_states.reshape(-1, target_hidden_states.shape[-1])
        if hidden.shape[0] != num_tokens:
            raise ValueError("MTP target hidden states and draft tokens must have identical token layouts")
        prepared_steps = None
        prepared_masks = False
        if gaudi_envs.VLLM_HPU_SPEC_OPTIMIZATION and batch_rows:
            if len(batch_rows) != last_indices.numel() or len(batch_rows) != len(last_positions_cpu):
                raise ValueError("MTP sampling rows, indices and logical positions must agree")
            prepared_steps, alive_steps = self.prepare_decode_steps(runner, batch_rows, last_positions_cpu)
            prepared_masks = bool(prepared_steps) and all(getattr(item, 'block_mapping', None) is not None
                                                          for item in prepared_steps)
        with mtp_phase('draft_head_forward'), mtp_phase('draft_head_step_0'):
            hidden = self.model(input_ids=token_ids,
                                positions=positions,
                                hidden_states=hidden,
                                inputs_embeds=None,
                                attn_metadata=metadata)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if not batch_rows:
            return torch.empty((0, self.num_speculative_tokens), dtype=torch.int32, device=self.device)
        if len(batch_rows) != last_indices.numel() or len(batch_rows) != len(last_positions_cpu):
            raise ValueError("MTP sampling rows, indices and logical positions must agree")
        hidden = hidden.index_select(0, last_indices.long())
        positions = select_mtp_positions(positions, last_indices, self.uses_mrope)
        logical_positions = list(last_positions_cpu)
        drafts = []
        for step in range(self.num_speculative_tokens):
            with mtp_phase('draft_logits_selection'), mtp_phase(f'draft_logits_step_{step}'):
                if prepared_steps is not None:
                    token, next_tokens, next_positions = self.model.compute_mtp_step(hidden, positions, alive_steps[step])
                else:
                    logits = self.model.compute_logits(hidden)
                    token = logits.argmax(dim=-1).to(torch.int32)
            if prepared_steps is None:
                alive = torch.tensor([position + 1 < self.max_model_len for position in logical_positions],
                                     dtype=torch.bool, device=self.device)
                token = torch.where(alive, token, -1)
            drafts.append(token)
            if step + 1 == self.num_speculative_tokens:
                break
            if prepared_steps is not None:
                metadata = prepared_steps[step]
                positions = next_positions
            else:
                logical_positions = [position + 1 for position in logical_positions]
                lengths = [int(position < self.max_model_len) for position in logical_positions]
                starts = [
                    position if length else 0 for position, length in zip(logical_positions, lengths, strict=True)
                ]
                metadata = self.decode_metadata(runner, batch_rows, starts, lengths, len(batch_rows), 1)
                positions = positions + 1
                positions = torch.where(alive.unsqueeze(0) if self.uses_mrope else alive, positions, 0)
                next_tokens = token.clamp_min(0).view(-1, 1)
            with mtp_phase('draft_head_forward'), mtp_phase(f'draft_head_step_{step + 1}'):
                hidden = self.model(input_ids=next_tokens,
                                    positions=positions,
                                    hidden_states=hidden,
                                    inputs_embeds=None,
                                    attn_metadata=metadata,
                                    _mtp_metadata_prepared=prepared_masks)
            hidden = hidden.reshape(-1, hidden.shape[-1])
        return torch.stack(drafts, dim=1)
