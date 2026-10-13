# SPDX-License-Identifier: Apache-2.0
"""Opt-in, deterministic-proposal FP32 TPC correction. No RNG in native code."""
import torch
from vllm_gaudi.v1.spec_decode.mtp_phase_profile import profile_stage


def load_native_library(path):
    if not path:
        raise ValueError('Native correction needs an explicitly qualified library')
    torch.ops.load_library(path)


def native_correct(draft_ids, cumulative_lengths, target_logits, bonus_ids, uniforms, noise):
    batch, width = draft_ids.shape
    vocab = target_logits.shape[-1]
    probs = target_logits.softmax(-1, dtype=torch.float32).view(batch, width, vocab).contiguous()
    ids = draft_ids.to(torch.int32).contiguous()
    cumulative = cumulative_lengths.to(torch.int32).contiguous()
    uniforms = uniforms.contiguous()
    noise = noise.contiguous()
    scores, recovered = torch.ops.custom_op.mtp_rejection_tiles(probs, ids, cumulative, uniforms, noise)
    return torch.ops.custom_op.mtp_rejection_pack(probs, ids, cumulative, uniforms,
                                                bonus_ids.to(torch.int32).view(batch, 1).contiguous(), scores, recovered)


_native_region = native_correct


def configure_native_region(compile_fn):
    global _native_region
    _native_region = compile_fn(native_correct)


@profile_stage('rejection_native_region')
def apply_native_correct(*args):
    return _native_region(*args)
