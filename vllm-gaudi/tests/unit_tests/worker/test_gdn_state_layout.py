# SPDX-License-Identifier: Apache-2.0
"""Contiguous checkpoint writes must be guarded by exact CPU ownership."""

import pytest
import torch

from vllm_gaudi.v1.worker.gdn_state_layout import has_contiguous_dflash_checkpoints


def _case(batch=1, capacity=4, tokens=8):
    indices = torch.zeros(5, batch, tokens, dtype=torch.int32)
    groups = {1: 0, 3: 1}
    for group, offset in groups.items():
        indices[group] = (torch.arange(batch * tokens, dtype=torch.int32).reshape(batch, tokens)
                          + offset * capacity * tokens + 1)
    return dict(state_indices=indices,
                group_offsets=groups,
                request_capacity=capacity,
                active_requests=batch,
                padded_requests=batch,
                tokens_per_request=tokens,
                slots_per_request=tokens,
                full_query=True)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("tokens", [8, 16])
def test_contiguous_checkpoint_ownership_accepts_exact_group_major_layout(batch, tokens):
    assert has_contiguous_dflash_checkpoints(**_case(batch, tokens=tokens))


@pytest.mark.parametrize("change", ["request_order", "checkpoint_order", "duplicate", "padding", "base_slot", "group"])
@pytest.mark.parametrize("tokens", [8, 16])
def test_contiguous_checkpoint_ownership_rejects_reordering_and_padding(change, tokens):
    args = _case(2, tokens=tokens)
    indices = args["state_indices"]
    if change == "request_order":
        indices[1] = indices[1].flip(0)
    elif change == "checkpoint_order":
        indices[3] = indices[3].flip(1)
    elif change == "duplicate":
        indices[3, 0, 1] = indices[3, 0, 0]
    elif change == "padding":
        indices[3, 1] = -1
    elif change == "base_slot":
        indices[1] += tokens
    elif change == "group":
        indices[3] = indices[1]
    assert not has_contiguous_dflash_checkpoints(**args)


@pytest.mark.parametrize("key,value", [
    ("full_query", False),
    ("active_requests", 0),
    ("active_requests", 1),
    ("padded_requests", 3),
    ("request_capacity", 1),
    ("tokens_per_request", 1),
    ("slots_per_request", 1),
    ("group_offsets", {}),
    ("group_offsets", {
        1: 0,
        3: 0
    }),
    ("group_offsets", {
        1: 0,
        5: 1
    }),
])
@pytest.mark.parametrize("tokens", [8, 16])
def test_contiguous_checkpoint_ownership_fails_closed_for_unsupported_metadata(key, value, tokens):
    args = _case(2, tokens=tokens)
    args[key] = value
    assert not has_contiguous_dflash_checkpoints(**args)


def test_contiguous_checkpoint_ownership_refuses_non_cpu_metadata():
    args = _case()
    args["state_indices"] = args["state_indices"].to("meta")
    with pytest.raises(ValueError, match="CPU"):
        has_contiguous_dflash_checkpoints(**args)


@pytest.mark.parametrize("tokens", [8, 16])
def test_model_runner_requires_opt_in_full_query_and_no_prefix_cache(tokens):
    from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

    runner = object.__new__(HPUModelRunner)
    runner._direct_gdn_state_enabled = True
    runner._direct_dflash_checkpoints_enabled = True
    runner._compact_gdn_enabled = True
    runner.use_prefix_caching = False
    runner._compact_gdn_group_ids = {1, 3}
    runner._compact_gdn_group_offset = {1: 0, 3: 1}
    runner._gdn_max_reqs = 4
    runner._gdn_state_slots_per_req = tokens
    indices = _case(2, tokens=tokens)["state_indices"]
    assert runner._can_use_direct_gdn_state(indices, 2, 2, tokens, True)
    assert not runner._can_use_direct_gdn_state(indices, 2, 2, tokens, False)
    runner.use_prefix_caching = True
    assert not runner._can_use_direct_gdn_state(indices, 2, 2, tokens, True)
    runner.use_prefix_caching = False
    runner._direct_dflash_checkpoints_enabled = False
    assert not runner._can_use_direct_gdn_state(indices, 2, 2, tokens, True)
