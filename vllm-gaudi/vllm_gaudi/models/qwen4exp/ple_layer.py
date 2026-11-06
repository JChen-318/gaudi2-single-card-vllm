# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PLE (Persistent Layer Embedding) layer for the Qwen4Exp HPU port.

Structure ported from upstream ``vllm/models/qwen4_exp/amd/ple_layer.py``:
a per-ple-layer module that owns the n-gram embedding (51B params across 128
shard tables, host-resident on this single-card deployment), a merged
``kv_proj`` projection (key stream of ``hc_count * hidden`` + query stream of
``hidden``), three grouped Gemma norms and a depthwise dilated conv1d that
together implement the PLE short-conv state update.

``forward`` runs the torch PLE runtime (n-gram hash, host-table lookup,
kv projection, gate, dilated short-conv). Shard tables are attached with
``attach_ngram`` after load; until the v1 worker owns the mamba cache, conv
history lives in a module-local buffer (one slot per request in the batch).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .config import Qwen4ExpTextConfig
from ._materialize import in_compile_only_mode as _in_compile_only_mode
from .h24 import h24_enabled as _h24_enabled
from .h29 import h29_enabled as _h29_enabled
from .ple_runtime import (
    PLENGramEmbedding,
    _spans_are_packed,
    apply_ple,
    apply_ple_fused,
    compose_ngram_ids,
    ple_gate,
    ple_short_conv,
)

__all__ = ["Qwen4ExpPLEGroupedNorm", "Qwen4ExpPLELayer"]


def _real_spans(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
) -> list[tuple[int, int, int]]:
    """``(slot, start, end)`` per request, ``-1`` bucket padding trimmed.

    ``slot`` is the padded-batch position, not the list index: the conv
    history and the n-gram context row are keyed by it and must stay stable
    when whole pad slots come and go between steps.

    H24-2 (``VLLM_QWEN4EXP_PLE_H24_SPANS``): both control tensors are read
    from the host twin :func:`control.bind_host_control` installed for this
    step (the model already hands the PLE layer a CPU ``input_ids``), so the
    per-layer ``.to("cpu")`` / ``.item()`` no longer drains the device queue.
    The tensor stays on the device only for a caller that never bound host
    control (a layer test or ``bench_layer_e2e``), which is the one case where
    the copy is unavoidable.
    """
    if query_start_loc is None:
        return [(0, 0, num_tokens)]
    host_qsl = host_ids = None
    if _h24_enabled("spans"):
        from .control import cpu_twin

        host_qsl = cpu_twin(query_start_loc, "query_start_loc")
        host_ids = cpu_twin(input_ids, "input_ids")
    if host_qsl is not None:
        loc = host_qsl.reshape(-1).tolist()
    else:
        loc = query_start_loc.detach().to("cpu").reshape(-1).tolist()
    ids = input_ids if host_ids is None else host_ids
    spans = []
    for i in range(len(loc) - 1):
        start, end = int(loc[i]), int(loc[i + 1])
        if end <= start:
            continue
        valid = int((ids[start:end] >= 0).sum().item())
        if valid > 0:
            spans.append((i, start, start + valid))
    return spans


class Qwen4ExpPLEGroupedNorm(nn.Module):
    """Grouped RMSNorm with Gemma (1 + w) affine over ``hc`` groups of H."""

    def __init__(
        self, hidden_size: int, eps: float, group_size: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float().unflatten(
            -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
        )
        variance = x.square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * (1.0 + self.weight.float().unflatten(
            -1, (normalized.shape[-2], self.group_size)))).flatten(-2).to(input_dtype)


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    """One PLE layer: n-gram embedding -> kv projection -> dilated conv.

    The embedding output feeds ``kv_proj`` whose merged output splits into
    the multi-stream key state (``hc_count * hidden``) and a single-stream
    query (``hidden``). The key stream is updated by a depthwise conv1d with
    dilation ``ngram_size``; its state length is
    ``(conv_kernel - 1) * ngram_size``.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        _ = use_sequence_parallel  # single-card port: no PLE sequence parallel
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id) if ple_dense_layer_id is not None else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        self.activation = "silu"

        self.eos_token_id = int(getattr(config, "eos_token_id", 0))
        self.heads_per_ngram = int(config.heads_per_ngram)
        # Host-resident n-gram tables. Not a submodule: the 51B shard tables
        # must not enter state_dict. Attached by the checkpoint loader.
        self.ngram: PLENGramEmbedding | None = None
        self._bringup_conv_state: torch.Tensor | None = None

        # Projection weights stay replicated (disable_tp) even when token rows
        # are sharded on multi-card deployments.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=self.model_config.dtype,
            quant_config=None,  # BF16 bring-up: quant stripped at load
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            self.model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=self.model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def attach_ngram(self, ngram: PLENGramEmbedding) -> None:
        """Attach host-resident shard tables for this PLE layer."""
        self.ngram = ngram

    def _conv_state_and_slots(
        self, num_reqs: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request conv history, ``[slots, window, C]`` (SD layout).

        The window is the *outer* of the two state axes because that is what
        ``get_state_shape`` announces (``MambaStateShapeCalculator``'s default
        SD orientation) and what the GDN conv pool uses, and because the H29
        conv kernel vectorizes along channels. Worker cache replaces this in
        Phase 3.
        """
        state = self._bringup_conv_state
        window = self.conv_state_len
        if (
            state is None
            or state.shape[0] < num_reqs
            or state.shape[1] != window
            or state.shape[2] != self.hc_hidden_size
            or state.dtype != dtype
            or state.device != device
        ):
            new = torch.zeros(
                max(num_reqs, 1),
                window,
                self.hc_hidden_size,
                dtype=dtype,
                device=device,
            )
            if (
                state is not None
                and state.shape[1] == new.shape[1]
                and state.shape[2] == new.shape[2]
            ):
                n = min(state.shape[0], new.shape[0])
                new[:n].copy_(state[:n].to(device=device, dtype=dtype))
            # Compile-only warmup compiles recipes without executing, so a
            # state tensor created there has no data. Caching it (as a plain
            # assignment would) poisons the first real forward: the state
            # reaches its graph as an input with empty tensor_data and dies
            # with ValidateSyncInputTensors. Build it for recipe compilation
            # but only cache a genuinely executed one (the Kimi-K25 guard
            # for the same trap).
            if not _in_compile_only_mode():
                self._bringup_conv_state = new
            state = new
        slots = torch.arange(num_reqs, device=device)
        return state, slots

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        spans: list[tuple[int, int, int]],
        ngram_context: torch.Tensor,
        pos_rows: torch.Tensor | None,
        conv_state: torch.Tensor,
    ) -> torch.Tensor | None:
        """One batched PLE pass for a packed span list, or None.

        H29-3. A packed list is ``[(i, i*L, (i+1)*L)]`` for ``i`` in
        ``range(B)`` — the shape of every decode step (``L == 1``, one token
        per request) and of every single-request prefill (``B == 1``). The
        per-span loop then costs one Python iteration, one ``kv_proj`` MME
        launch, one n-gram gather and two kernel launches *per request*; here
        the whole step is one of each, and at decode ``padded == real`` so the
        result is the layer output with no slice assignment at all.

        Fresh requests (position 0) zero their own slot first, which is what
        ``has_state=False`` means to ``ple_short_conv``.
        """
        if not _h29_enabled("batch"):
            return None
        length = _spans_are_packed(spans, hidden_states.shape[0])
        if not length:
            return None
        real = hidden_states.shape[0]
        # Row 0 of a request is a new sequence: its window starts at zeros.
        fresh = [
            pos_rows is not None and int(pos_rows[slot * length].item()) == 0
            for slot, _start, _end in spans
        ]
        if any(fresh):
            if all(fresh):
                conv_state[: len(spans)].zero_()
            else:
                for slot, is_fresh in zip((s for s, _a, _b in spans), fresh):
                    if is_fresh:
                        conv_state[slot].zero_()
        qsl = torch.arange(0, real + 1, length, dtype=torch.int32)
        ids = compose_ngram_ids(
            input_ids[:real],
            qsl,
            ngram_context[: len(spans)],
            self.ngram.layer_multipliers,
            self.ngram.head_vocab_sizes,
            self.ngram.head_offsets,
            self.eos_token_id,
            self.heads_per_ngram,
            to_device=False,
        )
        embeddings = self.ngram.lookup(
            ids, dtype=hidden_states.dtype, device=hidden_states.device
        )
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split([self.hc_hidden_size, self.hidden_size], dim=-1)
        hidden = hidden_states[:real]
        mixed = apply_ple_fused(
            hidden,
            key,
            value,
            conv_state,
            self.conv1d.weight,
            spans,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            eps=self.norm_key.variance_epsilon,
            dilation=self.short_conv_dilation,
        )
        if mixed is None:
            gated, conv_input = ple_gate(
                key,
                value,
                hidden,
                self.norm_key.weight,
                self.norm_query.weight,
                self.norm_conv.weight,
                self.norm_key.variance_epsilon,
            )
            activated = ple_short_conv(
                conv_input,
                qsl,
                conv_state,
                self.conv1d.weight,
                torch.arange(len(spans), dtype=torch.int64),
                dilation=self.short_conv_dilation,
            )
            mixed = (gated + activated) + hidden
        if real == hidden_states.shape[0]:
            return mixed
        out = hidden_states.clone()
        out[:real] = mixed
        return out

    @torch._dynamo.disable  # host n-gram lookup + per-slot conv history
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """N-gram lookup -> kv projection -> gate -> dilated short-conv.

        Runner bucket padding trims to real tokens: ``input_ids`` carry the
        ``-1`` sentinel after each request's real tokens and whole pad slots
        are all ``-1``. The n-gram hash must never see a pad id and the conv
        history must never absorb a pad row, so every stage runs per real
        span instead of over the padded whole.

        A span whose first position is 0 is a NEW request: its conv window
        starts at zeros (``has_state=False``). Without this the slot inherits
        the previous request's last tokens — a fresh request's first-token
        logits then depended on whatever ran before it.
        """
        if self.ngram is None:
            raise RuntimeError(
                "PLE n-gram tables are not attached; call attach_ngram() after load"
            )
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and {hidden_states.shape[0]}"
            )
        num_tokens = hidden_states.shape[0]
        spans = _real_spans(input_ids, query_start_loc, num_tokens)
        pos_rows = None
        if positions is not None:
            from .control import cpu_twin

            host = cpu_twin(positions, "positions")
            pos_rows = host if host is not None else positions
            if pos_rows.ndim > 1:
                pos_rows = pos_rows[0]
        padded_slots = int(query_start_loc.reshape(-1).numel() - 1) if query_start_loc is not None else 1
        conv_state, _ = self._conv_state_and_slots(
            padded_slots, hidden_states.dtype, hidden_states.device
        )
        # H29-3: one compose / lookup / kv_proj / gate / conv for the whole step
        # when every request carries the same token count (all decode steps and
        # every single-request prefill). Falls through to the per-span loop for
        # a mixed-length batch.
        batched = self._forward_packed(
            hidden_states, input_ids, spans, ngram_context, pos_rows, conv_state
        )
        if batched is not None:
            return batched
        out = hidden_states.clone()
        for slot, start, end in spans:
            n = end - start
            # No positions (legacy/test callers): assume a continuation.
            fresh = pos_rows is not None and int(pos_rows[start].item()) == 0
            has_state = torch.tensor([True]) if not fresh else None
            if fresh:
                conv_state[slot].zero_()
            # Control tensors stay on the host. apply_ple only reads them as
            # Python ints; copying them to HPU would drain the pipeline.
            qsl = torch.tensor([0, n], dtype=torch.int32)
            ids = compose_ngram_ids(
                input_ids[start:end],
                qsl,
                ngram_context[slot : slot + 1],
                self.ngram.layer_multipliers,
                self.ngram.head_vocab_sizes,
                self.ngram.head_offsets,
                self.eos_token_id,
                self.heads_per_ngram,
                to_device=False,
            )
            embeddings = self.ngram.lookup(
                ids, dtype=hidden_states.dtype, device=hidden_states.device
            )
            kv, _ = self.kv_proj(embeddings)
            key, value = kv.split([self.hc_hidden_size, self.hidden_size], dim=-1)
            slots = torch.tensor([slot], dtype=torch.int64)
            fused = apply_ple_fused(
                hidden_states[start:end],
                key,
                value,
                conv_state,
                self.conv1d.weight,
                [(slot, 0, n)],
                self.norm_key.weight,
                self.norm_query.weight,
                self.norm_conv.weight,
                eps=self.norm_key.variance_epsilon,
                dilation=self.short_conv_dilation,
                base_slot=slot,
            )
            if fused is None:
                fused = apply_ple(
                    hidden_states[start:end],
                    key,
                    value,
                    qsl,
                    conv_state,
                    self.conv1d.weight,
                    slots,
                    self.norm_key.weight,
                    self.norm_query.weight,
                    self.norm_conv.weight,
                    eps=self.norm_key.variance_epsilon,
                    dilation=self.short_conv_dilation,
                    has_state=has_state,
                )
            out[start:end] = fused
        return out
