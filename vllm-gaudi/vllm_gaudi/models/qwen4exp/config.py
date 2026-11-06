# SPDX-License-Identifier: Apache-2.0
"""Local Qwen4Exp configuration shim for the HPU port.

The checkpoint's ``config.json`` targets transformers 5.8-dev, which the
environment does not install. This shim parses the same JSON into plain
attribute objects exposing the fields the model port consumes (verified
against nvidia/Qwen3.8-Flash-Next-FP4-Local). It intentionally avoids the
transformers dependency; only the fields used by ``model.py`` are typed, the
rest are carried through verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _pick(source: dict[str, Any], name: str, default: Any) -> Any:
    value = source.get(name, default)
    if isinstance(value, dict) and name.endswith("_parameters"):
        # e.g. rope_parameters nested dicts stay as-is for getattr probing.
        return value
    return value


@dataclass
class Qwen4ExpTextConfig:
    """Language-model section of the Qwen4Exp config."""

    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    max_position_embeddings: int = 262144
    vocab_size: int = 248320
    tie_word_embeddings: bool = False
    # hybrid layering
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    # GDN (linear attention)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # indexer / QSA
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    output_gate_type: str = "sigmoid"
    norm_topk_prob: bool = True
    # hyperconnections
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE (n-gram embedding)
    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    split_ngram_parts: int = 128
    # MTP
    mtp_num_hidden_layers: int = 1
    # extras carried verbatim
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen4ExpTextConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extras"}
        kwargs = {k: data[k] for k in known if k in data}
        extras = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs, extras=extras)
        if not cfg.layer_types:
            cfg.layer_types = ["full_attention" if (i + 1) % cfg.full_attention_interval == 0
                               else "linear_attention"
                               for i in range(cfg.num_hidden_layers)]
        # Keep the serialized dialect verbatim: vendor layer accounting
        # (ModelConfig.get_num_layers_by_block_type) counts attention layers
        # by the literal "full_attention" marker. The model port accepts all
        # QSA spellings (see _QSA_LAYER_TYPES in model.py).
        return cfg

    # -- vLLM ModelConfig compatibility -----------------------------------
    model_type = "qwen4_exp"

    # transformers_utils.config.patch_rope_parameters calls these on every
    # text config carrying rope_parameters. The port consumes the checkpoint
    # dict verbatim via get_rope(), so both are intentional no-ops.
    def standardize_rope_params(self, *args, **kwargs) -> None:
        return None

    def validate_rope(self, *args, **kwargs) -> None:
        return None

    def get_text_config(self) -> "Qwen4ExpTextConfig":
        return self

    def __getattr__(self, name: str) -> Any:
        # Unknown checkpoint fields (hidden_act, rope locals, ...) resolved
        # from the verbatim extras instead of raising AttributeError.
        extras = self.__dict__.get("extras")
        if extras is not None and name in extras:
            return extras[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        """Plain view for transformers' ``GenerationConfig.from_model_config``.

        vLLM asks for a generation config when the checkpoint has no
        ``generation_config.json`` (the offline miniature checkpoint does not);
        transformers then calls ``model_config.to_dict()``, which this facade
        did not provide.
        """
        data = {key: value for key, value in self.__dict__.items() if key != "extras"}
        data.update(self.__dict__.get("extras") or {})
        return data


@dataclass
class Qwen4ExpVisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    # Present in the checkpoint; vLLM's multimodal/vision setup reads it.
    in_channels: int = 3
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    extras: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        # Checkpoint-only vision fields (deepstack_visual_indexes, ...) resolve
        # from the verbatim extras, matching the text config's facade.
        extras = self.__dict__.get("extras")
        if extras is not None and name in extras:
            return extras[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        data = {key: value for key, value in self.__dict__.items() if key != "extras"}
        data.update(self.__dict__.get("extras") or {})
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen4ExpVisionConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extras"}
        kwargs = {k: data[k] for k in known if k in data}
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extras=extras)


@dataclass
class Qwen4ExpConfig:
    """Root config mirroring the checkpoint's ``config.json``."""

    text: Qwen4ExpTextConfig = field(default_factory=Qwen4ExpTextConfig)
    vision: Qwen4ExpVisionConfig | None = None
    mtp: dict[str, Any] = field(default_factory=dict)
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    quantization_config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # Convenience passthroughs used by the model port.
    @property
    def hidden_size(self) -> int:
        return self.text.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text.num_hidden_layers

    @property
    def vocab_size(self) -> int:
        return self.text.vocab_size

    # transformers-compatible aliases used by vLLM model code.
    @property
    def text_config(self) -> Qwen4ExpTextConfig:
        return self.text

    @property
    def vision_config(self) -> Qwen4ExpVisionConfig | None:
        return self.vision

    # -- vLLM ModelConfig compatibility -----------------------------------
    model_type = "qwen4_exp"
    architectures = ["Qwen4ExpForConditionalGeneration"]
    torch_dtype = "bfloat16"

    def get_text_config(self) -> Qwen4ExpTextConfig:
        return self.text

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen4ExpConfig":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen4ExpConfig":
        text_data = dict(data.get("text_config") or {})
        # The checkpoint nests rope under text_config.rope_parameters; keep it
        # on the root AND on the text config (qwen3_next attention reads
        # config.rope_parameters off the text config).
        rope = text_data.get("rope_parameters", {}) or {}
        mtp = dict(text_data.pop("mtp", {}) or {})
        text_data.setdefault("rope_parameters", rope)
        vision_data = data.get("vision_config")
        quant = dict(data.get("quantization_config") or {})
        root_known = {"text_config", "vision_config", "quantization_config"}
        extras = {k: v for k, v in data.items() if k not in root_known}
        text = Qwen4ExpTextConfig.from_dict(text_data)
        return cls(
            text=text,
            vision=None if vision_data is None else Qwen4ExpVisionConfig.from_dict(vision_data),
            mtp=mtp,
            rope_parameters=dict(rope),
            quantization_config=quant,
            extras=extras,
        )
