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


# Max-free softmax (opt-in, VLLM_GAUDI_OMNI_SDPA_MAXFREE=1). On Gaudi2 the attention of a long DiT sequence is limited
# by the TPC (exp over N^2 scores), not the MME. With RMS-normalised q/k the scores are bounded, so each row can use the
# precomputable Cauchy-Schwarz bound b_i = |q_i| * max_j |k_j| * scale instead of its running max: no max pass, no
# rescaling, and the chunked Q.K^T -> exp -> P.V graph runs 20% faster than FusedSDPA (H3, 37,750 rows: 323 ms instead of
# about 400 ms per layer, error against FP32 0.6-0.7% instead of 0.9-1.0%). It needs b_i - rowmax_i well below ~80 so
# that exp(s - b) does not underflow: H3 layers 45, 48 and 49 hold outlier tokens (scores up to 9000) and must keep
# FusedSDPA, which VLLM_GAUDI_OMNI_SDPA_MAXFREE_SKIP_LAYERS lists (block indices of the DiT, comma separated).
HPU_SDPA_MAX_FREE = os.environ.get("VLLM_GAUDI_OMNI_SDPA_MAXFREE", "0") == "1"
HPU_SDPA_MAX_FREE_CHUNK = int(os.environ.get("VLLM_GAUDI_OMNI_SDPA_MAXFREE_CHUNK", "2560"))
HPU_SDPA_MAX_FREE_SKIP = frozenset(
    int(x) for x in os.environ.get("VLLM_GAUDI_OMNI_SDPA_MAXFREE_SKIP_LAYERS", "").replace(" ", "").split(",") if x)
HPU_SDPA_MAX_FREE_MIN_KV = 4096


def _max_free_chunk(qc: torch.Tensor, key: torch.Tensor, value: torch.Tensor, bound: torch.Tensor,
                    pad: float) -> torch.Tensor:
    scores = torch.matmul(qc, key.transpose(-1, -2))
    # The clamp only guards against an unexpected outlier (exp overflow); in range it changes nothing.
    probs = torch.exp(torch.clamp(scores - bound, max=30.0))
    row_sum = probs.sum(-1, keepdim=True, dtype=torch.float32)
    if pad:  # zero-padded keys score 0: each added exp(-b) to the sum and nothing to the numerator
        row_sum = row_sum - pad * torch.exp(-bound.float())
    return (torch.matmul(probs, value).float() / row_sum.clamp(min=1e-30)).to(qc.dtype)


_MAX_FREE_CHUNK_COMPILED = None


def _chunk_fn(device_type: str):
    """One compiled graph per query chunk. The chunk loop must stay out of any enclosing graph: compiled as a single
    graph (for example inside a regionally compiled DiT block) the graph compiler allocates every chunk's scores at
    once and fails or falls back to a slow schedule (H3: DiT forward 23 -> 47 s)."""
    global _MAX_FREE_CHUNK_COMPILED
    if device_type != "hpu":
        return _max_free_chunk
    if _MAX_FREE_CHUNK_COMPILED is None:
        _MAX_FREE_CHUNK_COMPILED = torch.compile(_max_free_chunk, backend="hpu_backend", dynamic=False)
    return _MAX_FREE_CHUNK_COMPILED


@torch.compiler.disable
def max_free_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
                       chunk: int = HPU_SDPA_MAX_FREE_CHUNK) -> torch.Tensor:
    """Softmax attention without a row max. query/key/value [B, H, L, D], same head count. Keys/values are
    zero-padded to a multiple of 128 (the unaligned length is slower on the MME) and the row sums are corrected
    exactly for the padding; queries are processed in chunks to bound the score memory (H*chunk*keys)."""
    q_scaled = query * scale
    key_max = key.float().norm(dim=-1).amax(-1, keepdim=True)  # [B, H, 1]
    bound = (q_scaled.float().norm(dim=-1) * key_max).to(query.dtype).unsqueeze(-1)  # [B, H, L, 1]
    pad = -key.shape[-2] % 128
    if pad:
        key = torch.nn.functional.pad(key, (0, 0, 0, pad))
        value = torch.nn.functional.pad(value, (0, 0, 0, pad))
    rows = query.shape[-2]
    run = _chunk_fn(query.device.type)
    outs = [
        run(q_scaled[..., i:i + chunk, :], key, value, bound[..., i:i + chunk, :], float(pad))
        for i in range(0, rows, chunk)
    ]
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)


def mark_max_free_layers(pipeline: object) -> int:
    """Enable the max-free path on the self-attention of every repeated DiT block except the skipped indices.
    Returns the number of enabled attention layers."""
    if not HPU_SDPA_MAX_FREE:
        return 0
    from torch import nn

    modules = list(pipeline.named_modules()) if isinstance(pipeline, nn.Module) else []
    if not modules:
        for attr in getattr(pipeline, "_dit_modules", None) or ("transformer", "transformer_2"):
            model = getattr(pipeline, attr, None)
            if isinstance(model, nn.Module):
                modules.extend(model.named_modules())
    enabled = 0
    for _, model in modules:
        block_names = set(getattr(model, "_repeated_blocks", None) or ())
        if not block_names:
            continue
        index = 0
        for _, block in model.named_modules():
            if type(block).__name__ not in block_names:
                continue
            for sub in block.modules():
                impl = getattr(sub, "attention", None)
                if isinstance(impl, HPUSDPAImpl):
                    impl._vllm_gaudi_max_free = index not in HPU_SDPA_MAX_FREE_SKIP
                    enabled += int(impl._vllm_gaudi_max_free)
            index += 1
    return enabled


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
        if (HPU_SDPA_MAX_FREE and getattr(self, "_vllm_gaudi_max_free", False) and attention_mask is None
                and not self.causal and kv_length >= HPU_SDPA_MAX_FREE_MIN_KV):
            key = key[:, :kv_length].permute(0, 2, 1, 3).contiguous()
            value = value[:, :kv_length].permute(0, 2, 1, 3).contiguous()
            if key.shape[1] != query.shape[1]:
                key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
                value = value.repeat_interleave(query.shape[1] // value.shape[1], dim=1)
            scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1]**-0.5
            output = max_free_attention(query, key, value, scale).permute(0, 2, 1, 3)
            if q_length < original_q_length:
                output = torch.cat((output, output.new_zeros(output.shape[0], original_q_length - q_length,
                                                             output.shape[2], output.shape[3])), dim=1)
            return output
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
           "kv_split_point", "mark_max_free_layers", "max_free_attention", "merge_kv_split"]
