# SPDX-License-Identifier: Apache-2.0
"""H23 switches: QSA 12-layer decode acceleration.

Four independent, measured rewrites of the QSA layer, each with its own
kill switch so a caller can A/B one item at a time. Every switch defaults to
the umbrella ``VLLM_QWEN4EXP_QSA_H23`` (default ``1``, on); set the umbrella
to ``0`` for the pre-H23 behavior of all four, or an item variable to
``0``/``1`` to override just that one.

=====================  ==========================  ==============================
item                   env                         what changes
=====================  ==========================  ==============================
23-1 attention          ``..._H23_ATTN``            dense_prefix_attention runs the
                                                   BF16 MME product on HPU
                                                   (fp32 mask/softmax kept)
23-2 host syncs         ``..._H23_SPANS``           one span/chunk-start host
                                                   extraction per step, no
                                                   device ``.item()``/``.max()``
                                                   assertion in the selection
23-3 tile replay        ``..._H23_TILE``            eager mode skips the 2048-block
                                                   window padding; desc-stable
                                                   argsort in the block top-k
23-4 projection/norm    ``..._H23_FUSE``            merged qkv+index GEMM and the
                                                   fused HPU RMSNorm
                                                   (``..._H23_MERGE=0`` keeps the
                                                   fused norm, drops the merge)
=====================  ==========================  ==============================

The switches are read live (tests set them with ``monkeypatch.setenv``), so
each read is a plain ``os.environ.get``.
"""
from __future__ import annotations

import os

__all__ = ["H23_ITEMS", "h23_enabled", "h23_item_env"]

# item -> per-item env override
H23_ITEMS = {
    "attn": "VLLM_QWEN4EXP_QSA_H23_ATTN",
    "spans": "VLLM_QWEN4EXP_QSA_H23_SPANS",
    "tile": "VLLM_QWEN4EXP_QSA_H23_TILE",
    "fuse": "VLLM_QWEN4EXP_QSA_H23_FUSE",
    # Sub-item of 23-4: the merged qkv+indexer GEMM. Defaults to the umbrella
    # like every other item, but it is *also* disabled whenever ``fuse`` is
    # off, so ``VLLM_QWEN4EXP_QSA_H23_MERGE=0`` isolates the fused RMSNorm.
    "merge": "VLLM_QWEN4EXP_QSA_H23_MERGE",
}

UMBRELLA = "VLLM_QWEN4EXP_QSA_H23"

_OFF = ("0", "false", "no", "off")
_ON = ("1", "true", "yes", "on")


def _parse(value: str, default: bool) -> bool:
    token = (value or "").strip().lower()
    if token in _OFF:
        return False
    if token in _ON:
        return True
    return default


def h23_item_env(item: str) -> str:
    """The env variable that overrides ``item`` (``KeyError`` if unknown)."""
    return H23_ITEMS[item]


def h23_enabled(item: str | None = None) -> bool:
    """Is the H23 rewrite (or one of its four items) enabled?

    ``item=None`` reads the umbrella only. Otherwise a non-empty per-item
    variable wins over the umbrella.
    """
    umbrella = _parse(os.environ.get(UMBRELLA, "1"), True)
    if item is None:
        return umbrella
    override = os.environ.get(H23_ITEMS[item], "")
    if (override or "").strip():
        return _parse(override, umbrella)
    return umbrella
