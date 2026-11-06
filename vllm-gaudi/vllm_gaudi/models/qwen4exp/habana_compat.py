# SPDX-License-Identifier: Apache-2.0
"""Work-around for a graph-compile crash in the Habana PyTorch bridge (seen on 1.24.1.482).

``habana_frameworks/torch/dynamo/_fx_to_jit_lowering.py``'s ``_get_stack_trace``
calls ``Path(torch.fx.graph._parse_stack_trace(st).file)`` without checking for
``None``. When torch cannot parse an FX node's stack trace (seen on a Dynamo
resume function of the GDN prefill graph, ``VLLM_QWEN4EXP_GDN_PREFILL_GRAPH=1``),
the whole compile fails with ``'NoneType' object has no attribute 'file'`` and
the engine dies on its first prompt. The stack trace is only a debug annotation
on the JIT node, so returning ``None`` on that failure changes no computation.

Applied by Qwen4Exp when it builds its text model; other models are not touched.
"""
from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def patch_fx_stack_trace() -> bool:
    """Make ``_get_stack_trace`` return ``None`` instead of raising. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from habana_frameworks.torch.dynamo import _fx_to_jit_lowering as lowering
    except ImportError:
        return False
    for name in dir(lowering):
        cls = getattr(lowering, name)
        if not isinstance(cls, type) or "_get_stack_trace" not in cls.__dict__:
            continue
        original = cls.__dict__["_get_stack_trace"]

        def safe_get_stack_trace(self, fx_node, _original=original):
            try:
                return _original(self, fx_node)
            except AttributeError:
                return None

        cls._get_stack_trace = safe_get_stack_trace
        _PATCHED = True
        logger.info("qwen4_exp: guarded %s.%s._get_stack_trace against unparsable stack traces",
                    lowering.__name__, name)
    return _PATCHED
