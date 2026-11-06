# SPDX-License-Identifier: Apache-2.0
"""H29 switches: one launch for the PLE gate and the PLE short conv.

Three independent rewrites of the one PLE layer, each with its own kill
switch so a caller can A/B one item at a time. The umbrella
``VLLM_QWEN4EXP_PLE_H29`` (default ``1``, on) covers all three; set it to
``0`` for the pre-H29 behavior. Each item variable overrides the umbrella for
that item only.

=====================  ==========================  ==============================
item                   env                         what changes
=====================  ==========================  ==============================
29-1 fused gate        ``..._H29_GATE``            the three grouped Gemma
                                                   RMSNorms, the BF16 gate dot,
                                                   the sigmoid and the conv-input
                                                   norm become ONE TPC launch
                                                   (``custom_qwen4exp_ple_gate``)
29-2 fused short conv  ``..._H29_CONV``            the dilated 4-tap conv, SiLU,
                                                   both residual adds, the
                                                   state roll and the per-span
                                                   loop become ONE TPC launch
                                                   (``custom_qwen4exp_ple_conv``)
29-3 batched layer     ``..._H29_BATCH``           one compose / lookup /
                                                   kv_proj / gate / conv for all
                                                   spans of the step instead of
                                                   a Python loop per request
=====================  ==========================  ==============================

Each item needs the one before it to be useful (the batched path calls the two
fused ops; with an op missing it falls back to torch per span), but they are
separately switchable so the bench can attribute each one.

The switches are read live (tests set them with ``monkeypatch.setenv``), so
each read is a plain ``os.environ.get``.
"""
from __future__ import annotations

import os

__all__ = ["H29_ITEMS", "h29_enabled", "h29_item_env"]

#: item -> per-item env override
H29_ITEMS = {
    "gate": "VLLM_QWEN4EXP_PLE_H29_GATE",
    "conv": "VLLM_QWEN4EXP_PLE_H29_CONV",
    "batch": "VLLM_QWEN4EXP_PLE_H29_BATCH",
}

UMBRELLA = "VLLM_QWEN4EXP_PLE_H29"

#: Items the umbrella covers (all of them).
UMBRELLA_ITEMS = ("gate", "conv", "batch")

_OFF = ("0", "false", "no", "off")
_ON = ("1", "true", "yes", "on")


def _parse(value: str, default: bool) -> bool:
    token = (value or "").strip().lower()
    if token in _OFF:
        return False
    if token in _ON:
        return True
    return default


def h29_item_env(item: str) -> str:
    """The env variable that overrides ``item`` (``KeyError`` if unknown)."""
    return H29_ITEMS[item]


def h29_enabled(item: str | None = None) -> bool:
    """Is the H29 rewrite (or one of its items) enabled?"""
    umbrella = _parse(os.environ.get(UMBRELLA, "1"), True)
    if item is None:
        return umbrella
    if item not in H29_ITEMS:
        raise KeyError(item)
    override = os.environ.get(H29_ITEMS[item], "")
    if (override or "").strip():
        return _parse(override, umbrella)
    if item not in UMBRELLA_ITEMS:
        return False
    return umbrella
