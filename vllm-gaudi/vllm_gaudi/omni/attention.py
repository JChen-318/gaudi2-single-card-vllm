# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense packed attention for vLLM-Omni diffusion models on Gaudi."""

from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl

HPU_SDPA_SOFTMAX_MODE = os.environ.get("VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE", "None")
if HPU_SDPA_SOFTMAX_MODE not in {"fp32", "None", "fast"}:
    raise ValueError("VLLM_GAUDI_OMNI_SDPA_SOFTMAX_MODE must be one of "
                     f"'fp32', 'None', or 'fast', got {HPU_SDPA_SOFTMAX_MODE!r}")

# FusedSDPA is about 30% slower when the key length is not a multiple of 128 (H3: 37,750 keys take 697 ms per
# layer, 37,632 take 521 ms), and its valid-length argument only exists for causal attention. Unmasked,
# non-causal attention over an unaligned key length therefore runs the aligned head of the keys and the short
# tail as two calls and merges them exactly through their row maxima and normalisers. 0 turns this off.
HPU_SDPA_KV_ALIGN = int(os.environ.get("VLLM_GAUDI_OMNI_SDPA_KV_ALIGN", "128"))
HPU_SDPA_KV_SPLIT_MIN = 4096


def kv_split_point(kv_length: int, align: int = HPU_SDPA_KV_ALIGN) -> int:
    """Length of the aligned key head to split at, or 0 to run one call."""
    if align <= 0 or kv_length < HPU_SDPA_KV_SPLIT_MIN or kv_length % align == 0:
        return 0
    return kv_length // align * align


def merge_kv_split(o1: torch.Tensor, m1: torch.Tensor, l1: torch.Tensor, o2: torch.Tensor, m2: torch.Tensor,
                   l2: torch.Tensor) -> torch.Tensor:
    """Exact softmax merge of two key blocks. ``o`` are each block's normalised outputs, ``m`` the row maxima and
    ``l`` the reciprocal row sums (the "fast" mode scales every ``l`` by the same constant, which cancels). The
    result is written as ``o1 + (o2 - o1) * b`` with ``b`` the second block's share: the tail block is short, so
    ``b`` is small and the update stays accurate in the output dtype without full-size FP32 temporaries."""
    m1, l1, m2, l2 = m1.float(), l1.float(), m2.float(), l2.float()
    mm = torch.maximum(m1, m2)
    w1 = torch.exp(m1 - mm) / l1
    w2 = torch.exp(m2 - mm) / l2
    share = (w2 / (w1 + w2)).to(o1.dtype)
    return torch.addcmul(o1, o2 - o1, share)


def additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean "may attend" mask -> additive bias (0 / -inf) in the query dtype. FusedSDPA takes a boolean mask on a
    much slower path for some lengths (Qwen-Image 2.1 decode, 4032 queries x 4096 keys: 70 ms instead of 4 ms per
    call); the equivalent bias is always on the fast path."""
    if mask.dtype != torch.bool:
        return mask
    return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))


def _sdpa_with_stats(query, key, value, scale):
    # Inputs must be contiguous: with a sliced view of a longer key tensor FusedSDPA returns wrong results for some
    # lengths (13% error measured at 36,864 of 37,750 keys).
    out, m, linv = torch.ops.hpu.sdpa_recomp_fwd(query, key, value, None, 0.0, scale, False, True,
                                                 HPU_SDPA_SOFTMAX_MODE, None, "right")[:3]
    return out, m, linv


class HPUSDPABackend(SDPABackend):
    """Gaudi FusedSDPA backend with mask-free H3 suffix-padding support."""

    accept_output_buffer = False
    supports_prefix_kv_slicing = True
    supported_platforms = ("oot", )

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "HPU_SDPA"

    @staticmethod
    def get_impl_cls() -> type[HPUSDPAImpl]:
        return HPUSDPAImpl


class HPUSDPAImpl(SDPAImpl):
    """Invoke Habana FusedSDPA without materializing a full padding mask."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if query.device.type != "hpu" or key.device.type != "hpu" or value.device.type != "hpu":
            raise RuntimeError("HPU_SDPA received non-HPU tensors; implicit host fallback is disabled")

        original_q_length = query.shape[1]
        q_length = original_q_length
        kv_length = key.shape[1]
        attention_mask = None
        if attn_metadata is not None:
            if attn_metadata.packed_padding is not None:
                q_length = int(attn_metadata.packed_padding.q_length)
                kv_length = int(attn_metadata.packed_padding.kv_length)
            elif "valid_kv_length" in attn_metadata.extra:
                kv_length = int(attn_metadata.extra["valid_kv_length"])
                q_length = min(q_length, kv_length)
            attention_mask = attn_metadata.attn_mask

        if attention_mask is not None and attention_mask.device.type != "hpu":
            raise RuntimeError("HPU_SDPA received a non-HPU attention mask; implicit host fallback is disabled")

        query = query[:, :q_length].permute(0, 2, 1, 3).contiguous()
        split = kv_split_point(kv_length) if attention_mask is None and not self.causal else 0
        if split:
            output = self._split_kv_forward(query, key, value, split, kv_length)
            if q_length < original_q_length:
                padding = output.new_zeros(output.shape[0], original_q_length - q_length, output.shape[2],
                                           output.shape[3])
                output = torch.cat((output, padding), dim=1)
            return output
        key = key[:, :kv_length].permute(0, 2, 1, 3).contiguous()
        value = value[:, :kv_length].permute(0, 2, 1, 3).contiguous()

        if query.shape[1] != key.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError("HPU FusedSDPA requires query heads to be a multiple of KV heads, "
                                 f"got {query.shape[1]} and {key.shape[1]}")
            repeat = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :kv_length]
            else:
                attention_mask = attention_mask[..., :q_length, :kv_length]
            attention_mask = additive_mask(attention_mask, query.dtype)

        from habana_frameworks.torch.hpex.kernels import FusedSDPA

        output = FusedSDPA.apply(
            query,
            key,
            value,
            attention_mask,
            0.0,
            self.causal,
            self.softmax_scale,
            HPU_SDPA_SOFTMAX_MODE,
            True,
        ).permute(0, 2, 1, 3)

        if q_length < original_q_length:
            padding = output.new_zeros(
                output.shape[0],
                original_q_length - q_length,
                output.shape[2],
                output.shape[3],
            )
            output = torch.cat((output, padding), dim=1)
        return output

    def _split_kv_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, split: int,
                          kv_length: int) -> torch.Tensor:
        """``query`` [B, Hq, q, D] contiguous; ``key``/``value`` [B, kv, Hkv, D] as given. Returns [B, q, Hq, D]."""

        def heads(t: torch.Tensor, start: int, stop: int) -> torch.Tensor:
            t = t[:, start:stop].permute(0, 2, 1, 3).contiguous()
            if t.shape[1] != query.shape[1]:
                if query.shape[1] % t.shape[1] != 0:
                    raise ValueError("HPU FusedSDPA requires query heads to be a multiple of KV heads, "
                                     f"got {query.shape[1]} and {t.shape[1]}")
                t = t.repeat_interleave(query.shape[1] // t.shape[1], dim=1)
            return t

        scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1]**-0.5
        o1, m1, l1 = _sdpa_with_stats(query, heads(key, 0, split), heads(value, 0, split), scale)
        o2, m2, l2 = _sdpa_with_stats(query, heads(key, split, kv_length), heads(value, split, kv_length), scale)
        return merge_kv_split(o1, m1, l1, o2, m2, l2).permute(0, 2, 1, 3)


__all__ = ["HPU_SDPA_KV_ALIGN", "HPU_SDPA_SOFTMAX_MODE", "HPUSDPABackend", "HPUSDPAImpl", "additive_mask",
           "kv_split_point", "merge_kv_split"]
