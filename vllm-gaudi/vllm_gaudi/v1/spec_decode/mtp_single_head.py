# SPDX-License-Identifier: Apache-2.0
"""Compile one original prediction head, never the recursive draft chain."""
import logging
from types import MethodType

from torch._dynamo.exc import Unsupported
from vllm_gaudi.v1.spec_decode.mtp_optimization_control import enabled

logger = logging.getLogger(__name__)


def _forward(self, *args, **kwargs):
    use_compiled = (enabled('single_head') and self._hpu_mtp_single_head_available
                    and self._hpu_mtp_decode_phase)
    function = (self._hpu_mtp_compiled_forward if use_compiled
                else self._hpu_mtp_original_forward)
    try:
        return function(*args, **kwargs)
    except Unsupported as exc:
        if not use_compiled:
            raise
        # Fullgraph tracing has not executed an HPU graph. Never catch device
        # execution, backend compilation, or arbitrary model exceptions here.
        self._hpu_mtp_single_head_available = False
        logger.warning('MTP fullgraph tracing unsupported; retain regional head: %s', exc)
        return self._hpu_mtp_original_forward(*args, **kwargs)


def configure_single_head(model, compile_fn):
    """Keep module names, weights, attention and cache ownership unchanged.

    Regional children remain available for a live baseline selection. Dynamo
    inlines their original computations into the enclosing tensor region.
    The original forward is called without changing its mathematical order.
    """
    inner = getattr(model, 'model', None)
    if (type(model).__name__ != 'Qwen3_5MTP' or inner is None
            or type(inner).__name__ != 'Qwen3_5MultiTokenPredictor'
            or inner.num_mtp_layers != 1
            or any(layer.use_attn_reduce_scatter_for_moe for layer in inner.layers)):
        return False
    if hasattr(inner, '_hpu_mtp_original_forward'):
        return True
    original = inner.forward
    compiled = compile_fn(original)
    # Plain attributes do not add another registered owner or rename KV layers.
    object.__setattr__(inner, '_hpu_mtp_original_forward', original)
    object.__setattr__(inner, '_hpu_mtp_compiled_forward', compiled)
    object.__setattr__(inner, '_hpu_mtp_single_head_available', True)
    object.__setattr__(inner, '_hpu_mtp_decode_phase', False)
    object.__setattr__(inner, 'forward', MethodType(_forward, inner))
    return True
