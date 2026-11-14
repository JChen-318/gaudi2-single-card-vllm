# SPDX-License-Identifier: Apache-2.0
"""Qwen DSpine draft using the Gaudi DFlash context/verification pipeline."""

import torch

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from vllm_gaudi.ops.dspine import DSpineCodeBridge, serial_refine, stable_topk_ids


class HpuDSpineModel(DFlashQwen3Model):
    def __init__(self, *, vllm_config, **kwargs):
        super().__init__(vllm_config=vllm_config, **kwargs)
        self.dspine_block_size = 1 + vllm_config.speculative_config.num_speculative_tokens
        self.selected_prefix_bridge = DSpineCodeBridge(self.config)

    def embed_input_ids(self, input_ids):
        return super().embed_input_ids(input_ids).to(torch.bfloat16)

    def forward(self, input_ids, positions, input_embeds=None):
        return self.forward_dspine(input_ids, positions, input_embeds)

    def forward_dspine(self, input_ids, positions, input_embeds=None):
        hidden = self.embed_input_ids(input_ids) if input_embeds is None else input_embeds
        hidden = hidden.reshape(-1, hidden.shape[-1])
        residual = None
        bridge = self.selected_prefix_bridge
        for index, layer in enumerate(self.layers):
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
            if index in bridge.active_layers:
                hidden, raw, rms, query = bridge(hidden, residual, input_ids, index, self.dspine_block_size)
                residual = None
        hidden = self.norm(hidden)
        # Explicit outputs carry refinement state across independently compiled
        # regions; a mutable Python stash would capture stale tensors on replay.
        return torch.cat((hidden.float(), raw.float(), rms, query.float()), dim=-1)


class HpuDSpineForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = HpuDSpineModel

    def __new__(cls, *, vllm_config, prefix=""):
        cfg = getattr(vllm_config.speculative_config.draft_model_config.hf_config, "dflash_config", None) or {}
        if not cfg.get("selected_prefix_code_bridge"):
            return DFlashQwen3ForCausalLM(vllm_config=vllm_config, prefix=prefix)
        return super().__new__(cls)

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.head_topk = int(self.config.code_bridge_second_pass["head_topk"])

    def load_weights(self, weights):
        bridge = self.model.selected_prefix_bridge
        destinations = dict(bridge.named_parameters())
        loaded = set()
        ordinary = []
        for name, tensor in weights:
            short = name.removeprefix("model.")
            if short.startswith("selected_prefix_bridge."):
                key = short.removeprefix("selected_prefix_bridge.")
                if key not in destinations or destinations[key].shape != tensor.shape:
                    raise ValueError(f"Unexpected DSpine bridge tensor: {name} {tensor.shape}")
                destinations[key].data.copy_(tensor)
                loaded.add(key)
            elif short in ("draft_block_causal_version", "last_write_refine_version"):
                if tensor.item() != 1:
                    raise ValueError(f"Unsupported DSpine version: {name}")
            else:
                ordinary.append((short, tensor))
        missing = set(destinations) - loaded
        if missing:
            raise ValueError(f"Missing DSpine bridge tensors: {sorted(missing)}")
        super().load_weights(ordinary)
        bridge.prepare()

    def compute_candidates(self, hidden_states):
        logits = self.compute_logits(hidden_states[..., :self.config.hidden_size].to(self.lm_head.weight.dtype))
        ids = stable_topk_ids(logits, self.head_topk)
        return ids, logits.gather(-1, ids.long())

    def combine_hidden_states(self, hidden_states):
        return super().combine_hidden_states(hidden_states.to(torch.bfloat16))

    def select_dspine_candidates(self, hidden_states, anchor_token_ids):
        b, steps, _ = hidden_states.shape
        h = self.config.hidden_size
        ids, _ = self.compute_candidates(hidden_states.reshape(-1, hidden_states.shape[-1]))
        ids = ids.to(torch.int32).reshape(b, steps, self.head_topk).sort(dim=-1).values
        raw = hidden_states[..., h:2 * h].to(torch.bfloat16)
        rms = hidden_states[..., 2 * h:2 * h + 1].contiguous()
        query = hidden_states[..., 2 * h + 1:].to(torch.bfloat16)
        return serial_refine(ids, raw, rms, query, anchor_token_ids, self.lm_head.weight,
                             self.model.norm.weight, self.model.selected_prefix_bridge)
