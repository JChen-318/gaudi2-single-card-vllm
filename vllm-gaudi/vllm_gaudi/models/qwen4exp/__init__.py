# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp (Qwen3.8-Flash-Next) HPU port modules.

Ported from the upstream ``vllm/models/qwen4_exp`` reference under
tools/qwen4exp_flashnext/reference/, adapted to the vendored vllm-core and
Gaudi plugin.
"""
# Set GC_KERNEL_PATH before any submodule (packed_moe / gdn_h14 /
# hyperconnection_amd / qsa / ...) can compile an HPU graph: importing the
# package is always the first step of importing any of them, and the HPU
# runtime resolves the kernel library at graph-compiler init, so a later
# set from an individual submodule may already be too late when a caller
# imported torch/habana first. The helper is dependency-free (no torch),
# idempotent, and preserves any existing GC_KERNEL_PATH entries.
from ._gc_kernel_path import ensure_gc_kernel_path

ensure_gc_kernel_path()

__all__ = ["ensure_gc_kernel_path"]
