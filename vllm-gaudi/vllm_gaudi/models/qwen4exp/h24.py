# SPDX-License-Identifier: Apache-2.0
"""H24 switches: PLE (persistent layer embedding) acceleration.

Four independent rewrites of the one PLE layer, each with its own kill
switch so a caller can A/B one item at a time. The umbrella
``VLLM_QWEN4EXP_PLE_H24`` (default ``1``, on) covers the three adopted
items; set it to ``0`` for the pre-H24 behavior of all of them. Each item
variable overrides the umbrella for that item only.

=====================  ==========================  ==============================
item                   env                         what changes
=====================  ==========================  ==============================
24-1 host gather       ``..._H24_GATHER``          the per-shard boolean-mask
                                                   loop becomes one stable
                                                   argsort + contiguous
                                                   per-shard segments
24-1b packed H2D       ``..._H24_H2D``             *measured and rejected*: move
                                                   90 B packed rows and dequant
                                                   on the HPU instead of moving
                                                   the 320 B decoded rows
24-2 grouped norm      ``..._H24_NORM``            the three grouped GemmaRMSNorm
                                                   calls go through the H13
                                                   ``hc_norm`` TPC kernel
24-2 host spans        ``..._H24_SPANS``           ``_real_spans`` reads the
                                                   host control twin instead of
                                                   draining the device queue
24-2 short conv        ``..._H24_CONV``            slice taps instead of index
                                                   tensors, cached fp32 weights,
                                                   single-rounding residual adds
=====================  ==========================  ==============================

``H2D`` is the only item whose built-in default is **off**: it lost at every
regime and it changes what crosses the host/device
boundary, which the host-offload accounting tests pin down. The umbrella never
turns it on; only the explicit item variable does.

The switches are read live (tests set them with ``monkeypatch.setenv``), so
each read is a plain ``os.environ.get``.
"""
from __future__ import annotations

import os

__all__ = ["H24_ITEMS", "h24_enabled", "h24_item_env"]

#: item -> per-item env override
H24_ITEMS = {
    "gather": "VLLM_QWEN4EXP_PLE_H24_GATHER",
    "h2d": "VLLM_QWEN4EXP_PLE_H24_H2D",
    "norm": "VLLM_QWEN4EXP_PLE_H24_NORM",
    "spans": "VLLM_QWEN4EXP_PLE_H24_SPANS",
    "conv": "VLLM_QWEN4EXP_PLE_H24_CONV",
}

UMBRELLA = "VLLM_QWEN4EXP_PLE_H24"

#: Items the umbrella covers (the adopted ones). ``h2d`` is measurement-only.
UMBRELLA_ITEMS = ("gather", "norm", "spans", "conv")

_OFF = ("0", "false", "no", "off")
_ON = ("1", "true", "yes", "on")


def _parse(value: str, default: bool) -> bool:
    token = (value or "").strip().lower()
    if token in _OFF:
        return False
    if token in _ON:
        return True
    return default


def h24_item_env(item: str) -> str:
    """The env variable that overrides ``item`` (``KeyError`` if unknown)."""
    return H24_ITEMS[item]


def h24_enabled(item: str | None = None) -> bool:
    """Is the H24 rewrite (or one of its items) enabled?

    ``item=None`` reads the umbrella only. Otherwise a non-empty per-item
    variable wins; without one, an adopted item follows the umbrella and the
    rejected ``h2d`` item stays off.
    """
    umbrella = _parse(os.environ.get(UMBRELLA, "1"), True)
    if item is None:
        return umbrella
    if item not in H24_ITEMS:
        raise KeyError(item)
    override = os.environ.get(H24_ITEMS[item], "")
    if (override or "").strip():
        return _parse(override, umbrella)
    if item not in UMBRELLA_ITEMS:
        return False
    return umbrella
