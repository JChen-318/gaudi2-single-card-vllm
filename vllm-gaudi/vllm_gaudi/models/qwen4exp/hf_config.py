# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""transformers-visible config for the Qwen4Exp HPU port.

The checkpoint's ``config.json`` targets transformers 5.8-dev, which this
environment does not install, so a bare ``AutoConfig.from_pretrained``
rejects ``model_type: qwen4_exp``. ``register_qwen4exp_configs`` registers a
thin ``PretrainedConfig`` facade so vLLM's standard config path works; every
attribute read delegates to the dataclass shim in ``config.py``, which stays
the single source of truth for the model port.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from transformers import PretrainedConfig

from .config import Qwen4ExpConfig

__all__ = ["Qwen4ExpHfConfig", "register_qwen4exp_configs"]


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    return value


class Qwen4ExpHfConfig(PretrainedConfig):
    """``PretrainedConfig`` facade over :class:`Qwen4ExpConfig`."""

    model_type = "qwen4_exp"

    def __init__(self, **kwargs: Any) -> None:
        text_data = kwargs.pop("text_config", None) or {}
        vision_data = kwargs.pop("vision_config", None)
        # BF16 bring-up: the loader dequantizes at load, so the FP4
        # quantization_config must not reach ModelConfig's quant resolver.
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)
        root = dict(kwargs)
        root["text_config"] = text_data
        root["vision_config"] = vision_data
        shim = Qwen4ExpConfig.from_dict(root)
        self._shim = shim
        self.text_config = shim.text
        self.vision_config = shim.vision
        self.hidden_size = shim.hidden_size
        self.num_hidden_layers = shim.num_hidden_layers
        self.vocab_size = shim.vocab_size
        self.quantization_config = None

    def __setattr__(self, key: str, value: Any) -> None:
        # vLLM's config loader re-attaches the checkpoint's FP4
        # quantization_config after __init__ (transformers_utils.config
        # get_config). The plain loader dequantizes at load, so the quant
        # resolver must never see it; the raw dict stays on the shim.
        if key == "quantization_config":
            value = None
        super().__setattr__(key, value)

    def get_text_config(self, decoder=None, encoder=None) -> Any:
        # transformers' dataclass validation calls this mid-``__init__``,
        # before the shim attributes exist; defer to the parent then.
        text = self.__dict__.get("text_config")
        if text is not None:
            return text
        return super().get_text_config(decoder=decoder, encoder=encoder)

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattr__(key)
        except AttributeError:
            pass
        shim = self.__dict__.get("_shim")
        if shim is not None and not key.startswith("_"):
            if hasattr(shim, key):
                return getattr(shim, key)
            text = shim.text
            if hasattr(text, key):
                return getattr(text, key)
        raise AttributeError(f"{type(self).__name__} has no attribute {key!r}")

    def to_dict(self) -> dict[str, Any]:
        # The nested shims are dataclasses, not PretrainedConfigs; serialize
        # them as plain dicts so repr/logging round-trips keep working.
        out = {
            k: _json_safe(v) for k, v in self.__dict__.items()
            if k not in ("_shim", "text_config", "vision_config")
        }
        shim = self.__dict__.get("_shim")
        if shim is not None:
            out["text_config"] = _plain(shim.text)
            if shim.vision is not None:
                out["vision_config"] = _plain(shim.vision)
        return out

    def to_diff_dict(self) -> dict[str, Any]:
        return self.to_dict()


def register_qwen4exp_configs() -> None:
    """Teach ``AutoConfig`` the checkpoint's ``model_type`` (idempotent)."""
    from transformers import AutoConfig

    AutoConfig.register("qwen4_exp", Qwen4ExpHfConfig, exist_ok=True)
