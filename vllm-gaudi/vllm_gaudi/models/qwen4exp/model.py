# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp model (Qwen3.8-Flash-Next) for Intel Gaudi.

Port of upstream ``vllm/models/qwen4_exp/amd/model.py`` adapted to the local
vllm-core vendor API and to a single-card HPU deployment:

- No HC/MoE sequence-parallel paths (TP=1 target).
- GDN linear attention resolves to ``HPUGatedDeltaNetAttention`` via the
  existing vllm-gaudi monkey-patch. The class is looked up at construction
  time, so the patch applies even if this module was imported first.
- QSA sparse attention is the torch indexer in ``qsa.py`` (exact above
  ``indexer_budget`` via a per-slot software cache). The paged HPU gather
  is still the hardware step; ``self.attn`` only reserves the KV spec.
- Quantization configs are stripped (BF16 bring-up): routed experts load
  through the plain NVFP4 planes produced by the CSF converter.
- ``MambaSpec``-style specs follow the local vendor protocol (no
  ``get_mamba_specs_from_config`` / state-copy-func registry here).
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _gdn_mod
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors

from .hyperconnection_amd import GatedResidual, HyperConnectionConfig
from .hyperconnection_base import is_hc_sequence_parallel_enabled
from .packed_moe import PackedRoutedExperts
from .ple_layer import Qwen4ExpPLELayer
from .ple_runtime import prepare_ple_inputs

logger = init_logger(__name__)

# Transformers v5.18 renamed `qwen_sparse_attention` to `indexed_attention`;
# checkpoints still serialize the plain `full_attention` marker, and vendor
# layer accounting counts attention layers by that literal string.
_QSA_LAYER_TYPES = ("qwen_sparse_attention", "indexed_attention", "full_attention")

_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint stores these projections separately; runtime packs each group
# into adjacent logical shards of a MergedColumnParallelLinear.
_EXTRA_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        "ple.key_proj": ("ple.kv_proj", 0),
        "ple.value_proj": ("ple.kv_proj", 1),
    }
)

try:  # dense fallback only if the torch QSA module failed to import
    from .qsa import Qwen4ExpQSAAttention  # noqa: F401

    _HAS_QSA = True
except ImportError:
    Qwen4ExpQSAAttention = None  # type: ignore[assignment,misc]
    _HAS_QSA = False


class Qwen4ExpSparseMoeBlock(nn.Module):
    """Sparse MoE with packed NVFP4 routed experts.

    The vendor block owns a ``FusedMoE`` whose ``w13_weight``/``w2_weight``
    are dense BF16: 48 layers x 512 experts x 3 projections is about 225 GiB,
    more than the 96 GiB HBM budget. This block keeps the same routing and
    shared-expert math but stores the routed planes packed and dequantizes
    only the selected experts per forward.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        _ = reduce_results  # single-card port: no results reduction
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.tp_size = 1
        self.n_routed_experts = int(config.num_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
        self.top_k = int(config.num_experts_per_tok)
        self.is_sequence_parallel = False

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = (
            Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=False,
                prefix=f"{prefix}.shared_expert",
            )
            if self.n_shared_experts
            else None
        )
        self.experts = PackedRoutedExperts(self.n_routed_experts)

    @torch._dynamo.disable  # custom TPC dequant is not in the capture allowlist
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from .packed_moe import (
            apply_routed_experts,
            apply_routed_experts_grouped,
            compact_expert_ids,
            grouped_from_ids,
            prefill_grouped_enabled,
            prefill_grouped_h18,
            prefill_grouped_h22,
            route_topk,
            use_grouped_slots,
        )

        orig_shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, orig_shape[-1])
        if hidden.shape[0] == 0:
            # Graph-capture and bucket warmup probe zero-token steps; there
            # is nothing to route and dequant_indexed refuses empty ids.
            return hidden_states.view(orig_shape)

        topk_ids, topk_weights = route_topk(hidden, self.gate.weight, self.top_k)
        if use_grouped_slots(hidden.shape[0], topk_ids.shape[-1]):
            # Decode: static top-k slots, one GEMM per projection, no host
            # round-trip. NVFP4 planes are dequantized exactly as before.
            out = grouped_from_ids(hidden, topk_ids, topk_weights, self.experts)
        else:
            # H18: the same three bucketed GEMMs, one count-ordered expert
            # chunk at a time (per-chunk cap + per-chunk BF16 peak, one
            # presence mask and one host transfer for the whole shape plan).
            # None -> compile-only warmup / non-HPU, keep the H9(a) path.
            # H22 first (projection-first dequant, one bucket for the layer,
            # per-chunk tight caps, bit-identical to H18), then H18, then the
            # H9(a) path below.
            out = None
            if prefill_grouped_enabled():
                out = prefill_grouped_h22(
                    hidden, topk_ids, topk_weights, self.experts)
            if out is None and prefill_grouped_enabled():
                out = prefill_grouped_h18(
                    hidden, topk_ids, topk_weights, self.experts)
            if out is None:
                # Presence mask, not torch.unique: the HPU unique kernel has
                # requested a multi-GiB workspace for a few thousand expert ids.
                unique, local = compact_expert_ids(topk_ids, self.n_routed_experts)
                gate_w = self.experts.dequant("gate_proj", unique)
                up_w = self.experts.dequant("up_proj", unique)
                down_w = self.experts.dequant("down_proj", unique)
                if prefill_grouped_enabled():
                    # Expert-major buckets + one bmm per projection (Strata's
                    # prefill shape) instead of one linear per selected expert.
                    out = apply_routed_experts_grouped(
                        hidden, local, topk_weights, gate_w, up_w, down_w)
                else:
                    out = apply_routed_experts(hidden, local, topk_weights, gate_w, up_w, down_w)
        if self.shared_expert is not None:
            # H15: merged gate_up+shared_gate GEMM -> fused SwiGLU/sigmoid
            # epilogue -> down GEMM (3 launches). Returns None (and the stock
            # MLP runs) when disabled, on CPU, or for uncovered shapes.
            from .packed_moe import shared_expert_forward_h15

            shared_out = shared_expert_forward_h15(self.shared_expert, hidden)
            out = out + (shared_out if shared_out is not None else self.shared_expert(hidden))
        return out.view(orig_shape)

    @property
    def is_fused_shared_expert_enabled(self) -> bool:
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):  # noqa: ANN201
        """Route ``experts.*`` to the packed store; delegate the rest."""
        rest: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if not self.experts.load_tensor(name, tensor):
                rest.append((name, tensor))
        loaded = AutoWeightsLoader(
            self, ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        ).load_weights(rest)
        return loaded | {f"experts.{i}" for i in range(self.n_routed_experts)}


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        config=None,
    ) -> None:
        super().__init__()
        # ``config`` lets a caller build a layer against a variant of the
        # text config (the MTP draft strips ple_layer_ids; nothing else).
        if config is None:
            config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        # Single-card port: HC sequence parallel is never enabled (TP=1).
        assert not is_hc_sequence_parallel_enabled(vllm_config)
        self.use_hc_sequence_parallel = False
        self.layer_idx = extract_layer_index(prefix)
        self.ple: Qwen4ExpPLELayer | None = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id_map[self.layer_idx + 1],
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            # Module attribute, not a bound import: qwen3_5.py replaces it
            # with HPUGatedDeltaNetAttention after this module may have loaded.
            self.linear_attn = _gdn_mod.QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            )
        elif layer_type in _QSA_LAYER_TYPES:
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if use_qsa and _HAS_QSA:
                self.self_attn = Qwen4ExpQSAAttention(  # type: ignore[misc]
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                if use_qsa:
                    logger.warning_once(
                        "Qwen4Exp QSA attention not ported yet; using dense "
                        "attention (exact for contexts <= indexer_budget=%s)",
                        getattr(config, "indexer_budget", 2048),
                    )
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen4ExpSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        # Embeddings can come back fp32 while HC weights are bf16.
        hc_dtype = attn_hc.hc_norm.weight.dtype
        if hidden_states.dtype != hc_dtype:
            hidden_states = hidden_states.to(dtype=hc_dtype)
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so pending HC state
            # must be materialized before the addition.
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                positions=positions,
            )

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type in _QSA_LAYER_TYPES:
            # query_start_loc splits the software QSA cache per request.
            # The dense fallback does not accept it.
            if _HAS_QSA:
                attn_out = self.self_attn(
                    hidden_states=block_input,
                    positions=positions,
                    query_start_loc=query_start_loc,
                )
            else:
                attn_out = self.self_attn(
                    hidden_states=block_input,
                    positions=positions,
                )
        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen4Exp routed experts through vLLM's EPLB protocol."""

    def set_moe_parameters(self, layers: Iterable[nn.Module] | None = None) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if layers is None:
            layers = self.model.layers
        for layer in layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen4ExpSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 0
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.use_hc_sequence_parallel = False
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], intermediate_size
        )

        self.hyper_connection_mixer: GatedResidual | None
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None
        # Batch-slot token history for PLE when the runner does not pass
        # ngram_context. Not a buffer: it is not a checkpoint tensor.
        self._ple_token_history: list[torch.Tensor] = []

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # Reshape before the host snapshot so QSA/PLE see the flat layout.
        if input_ids is not None and input_ids.ndim > 1:
            input_ids = input_ids.reshape(-1)
        if positions.ndim == 2 and positions.shape[0] != 3:
            positions = positions.reshape(-1)
        elif positions.ndim > 2 and positions.shape[0] == 3:
            positions = positions.reshape(3, -1)
        from .control import bind_host_control, reset_host_control

        ctrl_token = bind_host_control(positions, input_ids, query_start_loc)
        try:
            return self._forward_bound(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                query_start_loc,
                ngram_context,
                deepstack_input_embeds,
            )
        finally:
            reset_host_control(ctrl_token)

    def _forward_bound(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # The HPU runner passes bucketed [B, S] tokens. PLE and QSA want a
        # flat token axis. [3, T] (or [3, B, S]) is MRoPE and keeps the axis.
        if input_ids is not None and input_ids.ndim > 1:
            input_ids = input_ids.reshape(-1)
        if positions.ndim == 2 and positions.shape[0] != 3:
            positions = positions.reshape(-1)
        elif positions.ndim > 2 and positions.shape[0] == 3:
            positions = positions.reshape(3, -1)
        if inputs_embeds is not None and inputs_embeds.ndim == 3:
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
            if any(layer.ple is not None for layer in self.layers):
                if input_ids is None:
                    raise RuntimeError("PLE requires raw input_ids")
                from .control import host_control

                ctrl = host_control()
                ple_ids = ctrl.input_ids if ctrl is not None and ctrl.input_ids is not None else input_ids
                ple_pos = ctrl.positions if ctrl is not None and ctrl.positions is not None else positions
                ple_qsl = (
                    ctrl.query_start_loc
                    if ctrl is not None and ctrl.query_start_loc is not None
                    else query_start_loc
                )
                if query_start_loc is None or ngram_context is None:
                    query_start_loc, ngram_context = prepare_ple_inputs(
                        ple_ids,
                        ple_pos,
                        query_start_loc=ple_qsl,
                        ngram_context=ngram_context,
                        ngram_size=int(self.config.ngram_size),
                        eos_token_id=int(getattr(self.config, "eos_token_id", 0)),
                        history=self._ple_token_history,
                    )
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        from .control import cpu_twin

        layer_ids = cpu_twin(input_ids, "input_ids")
        if layer_ids is None:
            layer_ids = input_ids
        from .decode_graph import for_step as _decode_graph_for_step

        decode_graph = _decode_graph_for_step(self, hidden_states)
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            step = None
            if decode_graph is not None:
                step = decode_graph.run_layer(
                    layer,
                    hidden_states,
                    block_output,
                    injection,
                    positions=positions,
                    query_start_loc=query_start_loc,
                    input_ids=layer_ids,
                    ngram_context=ngram_context,
                )
            if step is None:
                step = layer(
                    hidden_states=hidden_states,
                    prev_block_output=block_output,
                    prev_injection=injection,
                    positions=positions,
                    input_ids=layer_ids,
                    query_start_loc=query_start_loc,
                    ngram_context=ngram_context,
                )
            hidden_states, block_output, injection = step
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                # Deepstack is an external addition to the materialized
                # multi-stream state and therefore terminates delayed combine.
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        # The final mixer consumes the last pending combine and returns both
        # the sampled single stream and the materialized multi-stream state.
        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        # Scheme A: the draft's first step consumes this pre-mixer multi
        # stream. Not a buffer — the MTP checkpoint does not store it.
        self._last_mtp_hidden = multi_hidden
        return sample_hidden_states

    def get_mtp_hidden(self) -> torch.Tensor | None:
        """Pre-final-mixer multi stream from the last forward, or None."""
        return getattr(self, "_last_mtp_hidden", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            enabled=False,  # shared-expert fusion disabled in BF16 bring-up
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = (
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        )
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={substr: None for substr in skip_substrs}
        )
        # The final HC mixer only exists on the last PP rank; earlier ranks
        # must drop its checkpoint weights instead of failing to place them.
        ignore_prefixes = (
            None
            if self.hyper_connection_mixer is not None
            else ["hyper_connection_mixer."]
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        return loaded


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),  # type: ignore[arg-type]
            ngram_context=kwargs.get("ngram_context"),  # type: ignore[arg-type]
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = hf_config.ple_conv_kernel_size
        short_conv_dilation = hf_config.ngram_size
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        hc_count = hf_config.hc_count
        hc_hidden_size = hf_config.hidden_size * hc_count
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=hc_hidden_size,
            conv_kernel=conv_state_len + 1,
            num_spec=num_spec,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_hidden(self) -> torch.Tensor | None:
        return self.model.get_mtp_hidden()

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None}
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)

    def attach_ple_tables(self, checkpoint_dir: str) -> int:
        """Load host-resident n-gram tables onto every PLE layer.

        The BF16 weight stream skips these (~27 GiB). Call after
        ``load_weights`` and before the first forward. Returns the number of
        layers attached.
        """
        from pathlib import Path

        from .ple_runtime import load_ple_ngram

        attached = 0
        for layer in self.model.layers:
            ple = getattr(layer, "ple", None)
            if ple is None:
                continue
            ple.attach_ngram(
                load_ple_ngram(
                    Path(checkpoint_dir),
                    layer_idx=ple.layer_idx,
                    heads_per_ngram=ple.heads_per_ngram,
                    # conv dilation == ngram_size in this architecture
                    # (transformers sets conv_dilation = config.ngram_size),
                    # and the table's head count needs the n-gram size.
                    ngram_size=ple.short_conv_dilation,
                )
            )
            attached += 1
        return attached


class Qwen4ExpForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen4ExpMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen4Exp language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "kv_proj": ["key_proj", "value_proj"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen4ExpForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = self._build_visual_tower(
                    config, vllm_config, quant_config, prefix
                )

        self.use_deepstack = (
            not self.language_model_only
            and bool(config.vision_config.deepstack_visual_indexes)
            and getattr(self, "visual", None) is not None
            and not isinstance(self.visual, nn.modules.container.ModuleDict)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.language_model.model.layers)

    def _build_visual_tower(
        self, config, vllm_config: VllmConfig, quant_config, prefix: str
    ) -> nn.Module:
        from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer

        return Qwen3_VisionTransformer(
            config.vision_config,
            norm_eps=config.text_config.rms_norm_eps,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen4Exp language_model_only does not accept multimodal embeddings"
            )
        from vllm.model_executor.models.utils import (
            _merge_multimodal_embeddings,
            _require_is_multimodal,
        )

        is_multimodal = _require_is_multimodal(is_multimodal)
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        deepstack_input_embeds = None
        if (
            inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self.use_deepstack
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if (
            inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self.use_deepstack
        ):
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen4ExpForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
