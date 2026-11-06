# SPDX-License-Identifier: Apache-2.0
"""H28 switches: QSA 12-layer step acceleration.

Two independent, measured rewrites of the QSA step. Each has its own item
variable so a caller can A/B one at a time; both default to the umbrella
``VLLM_QWEN4EXP_QSA_H28`` (default ``1``, on).

=========================  ===========================  ==============================
item                       env                          what changes
=========================  ===========================  ==============================
28-1 sparse decode         ``..._H28_SPARSE``           one untiled BF16 gather for
                                                        the gathered sparse attention
                                                        (no 1024-column cap) and the
                                                        fast route for it
28-2 MRoPE cos/sin         ``..._H28_ROPE``             one ``cos||sin`` composition
                                                        per step, shared by the main
                                                        q/k RoPE and the indexer
=========================  ===========================  ==============================

The switches are read live (tests set them with ``monkeypatch.setenv``), so
each read is a plain ``os.environ.get``.
"""
from __future__ import annotations

import os

__all__ = ["H28_ITEMS", "h28_enabled", "h28_item_env"]

# item -> per-item env override
H28_ITEMS = {
    "sparse": "VLLM_QWEN4EXP_QSA_H28_SPARSE",
    "rope": "VLLM_QWEN4EXP_QSA_H28_ROPE",
}

UMBRELLA = "VLLM_QWEN4EXP_QSA_H28"

_OFF = ("0", "false", "no", "off")
_ON = ("1", "true", "yes", "on")


def _parse(value: str, default: bool) -> bool:
    token = (value or "").strip().lower()
    if token in _OFF:
        return False
    if token in _ON:
        return True
    return default


def h28_item_env(item: str) -> str:
    """The env variable that overrides ``item`` (``KeyError`` if unknown)."""
    return H28_ITEMS[item]


def h28_enabled(item: str | None = None) -> bool:
    """Is the H28 rewrite (or one of its two items) enabled?

    ``item=None`` reads the umbrella only. Otherwise a non-empty per-item
    variable wins over the umbrella.
    """
    umbrella = _parse(os.environ.get(UMBRELLA, "1"), True)
    if item is None:
        return umbrella
    override = os.environ.get(H28_ITEMS[item], "")
    if (override or "").strip():
        return _parse(override, umbrella)
    return umbrella
