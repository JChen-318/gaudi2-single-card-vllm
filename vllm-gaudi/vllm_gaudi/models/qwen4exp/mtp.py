# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MTP draft head for the Gaudi port.

One indexed-attention decoder layer. The checkpoint has no embed or LM head
under ``mtp.`` — those are shared from the target via ``share_embeddings``.
Fusion is residual-linear-shared, not the Qwen3.5 ``Linear(2H, H)`` concat:

    embed = fc_embedding(GemmaRMSNorm(token))
    hidden = fc_hidden(GemmaRMSNorm(multi_stream))   # per HC branch
    draft_in = (embed + hidden).flatten(HC)

The target stashes the pre-final-mixer multi stream on
``Qwen4ExpModel._last_mtp_hidden``. Later draft steps reuse this module's
own multi-stream return. Paged draft KV and the HPU proposer stay Phase 3.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen3_next import Qwen3NextModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .hyperconnection_amd import GatedResidual, HyperConnectionConfig
from .model import (
    _EXTRA_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
)

__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor", "remap_mtp_weight_name"]


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def remap_mtp_weight_name(name: str) -> str | None:
    """Map checkpoint ``mtp.*`` paths onto this draft module.

    ``mtp.layers.0.*`` stays layer 0 (the ModuleList index). Attention layer
    names still carry the ``mtp.`` prefix, so they do not collide with the
    target's ``model.layers.0``. Embed and LM head are absent from the MTP
    group and are shared from the target instead.
    """
    for checkpoint_prefix in ("model.language_model.", "language_model."):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break
    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


class Qwen4ExpMultiTokenPredictor(nn.Module):
    """One-step draft trunk: fusion, indexed-attention layer, HC mixer."""

    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.num_mtp_layers = int(getattr(config, "mtp_num_hidden_layers", 1) or 1)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        # BF16 bring-up: fusion linears are unquantized even if the trunk is FP4.
        self.fc_embedding = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.fc_hidden",
        )
        # Checkpoint stores the draft layer at layers.0, and that layer is
        # indexed attention (q/k/v + indexer), not the trunk's layer-0 GDN.
        # PLE stays off: the mtp.* checkpoint group ships no PLE tables, so
        # the draft never builds one even when the trunk's ple_layer_ids
        # would match the draft's own layer index (ple_layer_ids=[1] puts
        # the trunk's PLE on layer 0, which is also the draft's index).
        draft_config = (
            config if not config.ple_layer_ids
            else dataclasses.replace(config, ple_layer_ids=[])
        )
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    vllm_config,
                    layer_type="indexed_attention",
                    prefix=f"{prefix}.layers.{idx}",
                    config=draft_config,
                )
                for idx in range(self.num_mtp_layers)
            ]
        )
        if any(layer.ple is not None for layer in self.layers):
            raise RuntimeError("MTP draft layers must not own a PLE table")

        eps = float(config.rms_norm_eps)
        self.pre_fc_norm_embedding = GemmaRMSNorm(self.hidden_size, eps=eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(self.hidden_size * self.hc_count, eps=eps)
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=int(config.hc_lowrank),
            rms_norm_eps=eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def fuse(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return the HC-width draft input ``[T, hc_count * H]``."""
        if hidden_states.shape[-1] != self.hidden_size * self.hc_count:
            raise ValueError(
                "MTP hidden must be the pre-mixer multi stream "
                f"[T, {self.hc_count * self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        # Embeddings may come back in fp32 while the draft weights are bf16.
        dtype = self.fc_embedding.weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=dtype)
        hidden_states = hidden_states.to(dtype=dtype)
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        inputs_embeds = _linear(self.fc_embedding, inputs_embeds)
        num_tokens = hidden_states.shape[0]
        hidden_states = hidden_states.view(num_tokens, self.hc_count, self.hidden_size)
        hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(
            num_tokens, self.hc_count, self.hidden_size
        )
        hidden_states = _linear(self.fc_hidden, hidden_states)
        return (inputs_embeds.unsqueeze(-2) + hidden_states).flatten(-2)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if hidden_states is None:
                raise ValueError("MTP draft requires the target multi-stream hidden")
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("MTP draft requires input_ids or inputs_embeds")
                inputs_embeds = self.embed_input_ids(input_ids)
            hidden_states = self.fuse(inputs_embeds, hidden_states)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        layer = self.layers[spec_step_idx % self.num_mtp_layers]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        multi_hidden, sample_hidden, _ = self.hyper_connection_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        return sample_hidden, multi_hidden


class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
    """Standalone draft model. Register as ``Qwen4ExpMTP``."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = Qwen4ExpMultiTokenPredictor.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    int(config.vocab_size),
                    int(config.hidden_size),
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(int(config.vocab_size))
        self.set_moe_parameters(self.model.layers)

    def share_embeddings(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        """Alias the target embed and LM head. The MTP checkpoint has neither."""
        self.model.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        _ = spec_step_idx
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remapped():
            for name, weight in weights:
                mapped = remap_mtp_weight_name(name)
                if mapped is not None:
                    yield mapped, weight

        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(remapped(), mapper=self.hf_to_vllm_mapper)
