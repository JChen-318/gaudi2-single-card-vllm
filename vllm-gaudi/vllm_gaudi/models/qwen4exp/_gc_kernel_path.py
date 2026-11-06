# SPDX-License-Identifier: Apache-2.0
"""Shared early ``GC_KERNEL_PATH`` setup for the Qwen4Exp TPC kernels.

SynapseAI resolves the TPC kernel library named by ``GC_KERNEL_PATH`` when
the HPU graph compiler first initializes, not when a Python module later
happens to run. A script that imports ``torch`` / ``habana_frameworks``
before importing a Qwen4Exp module -- or that imports ``gdn_h14`` /
``hyperconnection_amd`` (which historically set nothing) before
``packed_moe`` -- can therefore compile its first HPU graph against the
stock TPC library, which does not carry the custom Qwen4Exp GUIDs.

Calling :func:`ensure_gc_kernel_path` at the top of every Qwen4Exp module
that can reach an HPU graph compile (before ``torch`` / vLLM imports in
those modules where practical, and always before any kernel launch) makes
the custom library visible regardless of import order. This module is
deliberately dependency-free (no ``torch``) so it can run that early and
so importing it through the package ``__init__`` cannot create a cycle.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["ensure_gc_kernel_path", "qwen4exp_kernel_library"]

_KERNEL_NAME = "libqwen4exp_gaudi2_kernels.so"


def qwen4exp_kernel_library() -> Path:
    """Path of the built Qwen4Exp TPC kernel library for this checkout.

    ``_gc_kernel_path.py`` lives at
    ``<repo>/vllm-gaudi/vllm_gaudi/models/qwen4exp/``; the CMake build sits
    in ``<repo>/vllm-gaudi/csrc/qwen4exp/build/``.
    """
    return Path(__file__).resolve().parents[3] / "csrc" / "qwen4exp" / "build" / _KERNEL_NAME


def ensure_gc_kernel_path() -> str:
    """Prepend the Qwen4Exp kernel library to ``GC_KERNEL_PATH``.

    Idempotent: if the library path is already one of the
    ``os.pathsep``-separated entries it is left in place (never
    duplicated). Existing entries (e.g. the stock
    ``/usr/lib/habanalabs/libtpc_kernels.so``) are preserved after it.
    If the built library does not exist yet, the environment is left
    untouched so a host/CPU-only import does not point SynapseAI at a
    non-existent file. Returns the resulting ``GC_KERNEL_PATH`` value.
    """
    kernel = str(qwen4exp_kernel_library())
    current = os.environ.get("GC_KERNEL_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if kernel in entries:
        return current
    if not Path(kernel).is_file():
        return current
    os.environ["GC_KERNEL_PATH"] = os.pathsep.join([kernel, *entries])
    return os.environ["GC_KERNEL_PATH"]
