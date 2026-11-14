# SPDX-License-Identifier: Apache-2.0
"""Bridge isolation and serial refinement against scalar checkpoint equations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

path = Path(__file__).resolve().parents[1] / "vllm-gaudi/vllm_gaudi/ops/dspine.py"
spec = importlib.util.spec_from_file_location("dspine_ops", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def test_topk_boundary_ties_choose_lowest_token_ids():
    logits = torch.zeros(3, 32).bfloat16()
    logits[1, [21, 30]] = 2
    logits[2, [5, 11, 19, 23]] = 1
    expected = torch.tensor([[0, 1, 2, 3], [0, 1, 21, 30], [5, 11, 19, 23]], dtype=torch.int32)
    torch.testing.assert_close(ops.stable_topk_ids(logits, 4), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 7, 1000), (2, 248320)])
def test_chunked_topk_matches_full_stable_sort(shape):
    generator = torch.Generator().manual_seed(92)
    logits = torch.randn(shape, generator=generator).bfloat16()
    logits.reshape(-1, shape[-1])[0].zero_()
    expected = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :16]
    expected = expected.to(torch.int32).sort(dim=-1).values
    torch.testing.assert_close(ops.stable_topk_ids(logits, 16), expected, rtol=0, atol=0)


def make_bridge():
    torch.manual_seed(73)
    config = SimpleNamespace(hidden_size=32, vocab_size=64, num_hidden_layers=5, dflash_config={
        "selected_prefix_code_bridge": {"code_size": 16, "rank": 8, "eps": 1e-6},
        "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
    })
    bridge = ops.DSpineCodeBridge(config)
    with torch.no_grad():
        for p in bridge.parameters():
            if p.is_floating_point():
                p.normal_(0, 0.15)
        bridge.out.weight.zero_()
        bridge.independent_write_version.fill_(1)
        bridge.terminal_disabled_version.fill_(1)
        bridge.prepare()
    return bridge


def test_bridge_anchor_and_request_isolation():
    bridge = make_bridge()
    for t in (8, 16):
        hidden = torch.randn(2 * t, 32).bfloat16()
        residual = torch.randn_like(hidden)
        ids = torch.zeros((2, t), dtype=torch.long)
        ids[:, 0] = torch.tensor([3, 17])
        out, raw, rms, query = bridge(hidden, residual, ids, 4, t)
        torch.testing.assert_close(out[::t], raw[::t], rtol=0, atol=0)
        other = hidden.clone()
        other[t:] *= 4
        changed = bridge(other, residual, ids, 4, t)[0]
        torch.testing.assert_close(out[:t], changed[:t], rtol=0, atol=0)
        for batch in range(2):
            begin, end = batch * t, (batch + 1) * t
            one = bridge(hidden[begin:end], residual[begin:end], ids[batch:batch + 1], 4, t)
            for actual, expected in zip((out, raw, rms, query), one):
                torch.testing.assert_close(actual[begin:end], expected, rtol=0, atol=0)


def test_serial_refinement_matches_scalar_tables():
    bridge = make_bridge()
    lm = torch.randn(64, 32).bfloat16()
    norm = torch.randn(32).bfloat16()
    tg = ((lm.float() * norm.float()).bfloat16() @ bridge.feature_write_weights[-1]).float()
    tsv = F.linear((4 * bridge.codebook.float()).bfloat16(), bridge.packed_weight[8:]).float()
    for steps in (7, 15):
        ids = torch.rand(2, steps, 64).topk(4, dim=-1).indices.sort(-1).values
        raw = torch.randn(2, steps, 32).bfloat16()
        rms = torch.rand(2, steps, 1)
        query = torch.randn(2, steps, 8).bfloat16()
        anchors = torch.tensor([3, 17])
        actual = ops.serial_refine(ids, raw, rms, query, anchors, lm, norm, bridge)
        expected = torch.empty_like(actual)
        for b in range(2):
            previous = anchors[b]
            for s in range(steps):
                sender, value = tsv[previous].split(8)
                message = torch.sigmoid(query[b, s].float() + sender + bridge.gate.bias.float()) * value
                candidates = ids[b, s]
                base = (lm[candidates].float() * norm.float() * raw[b, s].float()).sum(-1)
                scores = base + rms[b, s] * (tg[candidates] * message).sum(-1)
                previous = candidates[scores.argmax()]
                expected[b, s] = previous
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
