# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization lookup tables shared by the Qwen4Exp loader (torch port of
the validated numpy math in ``tools/qwen4exp_flashnext/csf.py``)."""

from __future__ import annotations

import torch

__all__ = ["e2m1_value_lut", "e4m3_value_lut"]


def e2m1_value_lut() -> torch.Tensor:
    """[16] float32 table: E2M1 code (sign << 3 | magnitude) -> value."""
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    lut = mags.repeat(2)
    lut[8:] = -lut[8:]
    return lut


def e4m3_value_lut() -> torch.Tensor:
    """[256] float32 table: E4M3FN byte -> value (ported from csf.py)."""
    codes = torch.arange(256, dtype=torch.int32)
    sign = torch.where(codes & 0x80 != 0, -1.0, 1.0)
    e = (codes >> 3) & 0x0F
    m = codes & 0x07
    normal = sign * (1.0 + m.to(torch.float32) / 8.0) * torch.exp2(e.to(torch.float32) - 7.0)
    subnormal = sign * (m.to(torch.float32) / 8.0) * torch.exp2(torch.tensor(-6.0))
    value = torch.where(e == 0, subnormal, normal)
    # E4M3FN: 0x7F/0xFF are the only NaN codes (no infinities).
    value = torch.where((codes & 0x7F) == 0x7F, float("nan"), value)
    return value
