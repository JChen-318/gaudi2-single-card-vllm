# SPDX-License-Identifier: Apache-2.0
"""Bounded, read-only logits/address snapshots for the two authorized requests."""
from __future__ import annotations

import json
import os

import torch

_COUNTS = {}
_REQUESTS = {}


def capture(runner, decode_data, logits, metadata, request_ids):
    marker = os.environ.get("DFLASH2_ROUTING_TRACE_MARKER")
    if not marker or not os.path.isfile(marker):
        return
    row = {"kind": "routing_logits", "pid": os.getpid()}
    try:
        with open(marker, encoding="utf-8") as stream:
            label = stream.read().strip()
        if label not in ("prefix", "gold") or _COUNTS.get(label, 0) >= 8:
            return
        if len(request_ids) != 1:
            raise ValueError("Expected one diagnostic request")
        req_id = request_ids[0]
        if label in _REQUESTS and _REQUESTS[label] != req_id:
            return
        _REQUESTS[label] = req_id
        _COUNTS[label] = _COUNTS.get(label, 0) + 1
        row.update(label=label, step=_COUNTS[label], request_id=req_id)
        request = runner.requests[req_id]
        width = int(metadata.num_draft_tokens[0]) + 1
        raw = logits[:width].detach().to("cpu").float()
        inputs = decode_data.token_ids.detach().to("cpu").flatten()[:width].tolist()
        positions = decode_data.position_ids.detach().to("cpu")
        axis = positions[0] if positions.ndim == 2 and positions.shape[0] == 3 else positions.flatten()
        pos = axis[:width].tolist()
        params = request.sampling_params
        filtered, order = (raw / params.temperature).sort(dim=-1, descending=False)
        if 0 < params.top_k < raw.shape[-1]:
            threshold = filtered[:, -params.top_k].unsqueeze(-1)
            filtered.masked_fill_(filtered < threshold, -float("inf"))
        if params.top_p < 1:
            remove = filtered.softmax(-1).cumsum(-1) <= 1 - params.top_p
            remove[:, -1] = False
            filtered.masked_fill_(remove, -float("inf"))
        probs = torch.empty_like(filtered).scatter_(1, order, filtered).softmax(-1)
        values, ids = raw.topk(5)
        pvalues, pids = probs.topk(5)
        row.update(width=width, draft_lengths=metadata.num_draft_tokens, input_ids=inputs, positions=pos,
                   temperature=params.temperature, top_k=params.top_k, top_p=params.top_p,
                   raw_nonfinite=int((~raw.isfinite()).sum()),
                   raw_top={"ids": ids.tolist(), "values": values.tolist()},
                   cpu_filtered_top={"ids": pids.tolist(), "values": pvalues.tolist()},
                   zero_prob=probs[:, 15].tolist(), group_addresses={})
        attn = decode_data.attn_metadata
        req_index = runner.input_batch.req_id_to_index[req_id]
        for gid in runner._legacy_attention_group_ids:
            group = attn.attention_groups[gid]
            observed = group.slot_mapping.detach().to("cpu").flatten()[:width].tolist()
            table = runner.input_batch.block_table[gid].get_cpu_tensor()[req_index]
            expected = [runner._resolve_block(int(table[int(p)//runner.attn_block_size]))*runner.attn_block_size
                        + int(p)%runner.attn_block_size for p in pos]
            row["group_addresses"][str(gid)] = {"slots": observed, "expected_slots": expected,
                                                 "match": observed == expected}
        row["group_slots_disjoint"] = len({slot for group in row["group_addresses"].values()
                                             for slot in group["slots"]}) == width*len(row["group_addresses"])
        if label == "prefix":
            with open(os.environ["DFLASH2_ROUTING_REFERENCE"], encoding="utf-8") as stream:
                reference = json.load(stream)
            committed = list(request.prompt_token_ids) + list(request.output_token_ids)
            matches = [index for index, p in enumerate(pos)
                       if committed[:int(pos[0])] + inputs[:index+1] == reference["prefix_ids"]]
            row["matched_reference_rows"] = matches
            if matches:
                path = os.environ["DFLASH2_ROUTING_TENSOR"]
                if not os.path.exists(path):
                    torch.save(raw[matches[0]].clone(), path)
                row["matched_logits_at_reference_ids"] = raw[matches[0], reference["reference_raw_top"]["ids"]].tolist()
    except Exception as exc:
        row["error"] = repr(exc)
    try:
        with open(os.environ["DFLASH2_ROUTING_TRACE_LOG"], "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    except Exception:
        import traceback
        traceback.print_exc()
