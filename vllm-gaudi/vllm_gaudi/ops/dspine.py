# SPDX-License-Identifier: Apache-2.0
"""DSpine BF16 bridge and deterministic serial refinement on torch devices.

Port of Tencent/DSpine's prepare_read, normalize_codes, feature_message,
feature_finish and serial_table equations (Apache-2.0). No CUDA/Triton ops.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _topk_values(scores, k):
    """Exact top-k values from the k chunks with the largest maxima."""
    rows, vocab = scores.shape
    chunk = 256
    chunks = (vocab + chunk - 1) // chunk
    if chunks * chunk != vocab:
        scores = F.pad(scores, (0, chunks * chunk - vocab), value=float("-inf"))
    blocks = scores.reshape(rows, chunks, chunk)
    picked = min(k, chunks)
    chunk_ids = blocks.amax(-1).topk(picked, dim=-1).indices
    values = blocks.gather(1, chunk_ids.unsqueeze(-1).expand(rows, picked, chunk)).reshape(rows, -1)
    return values.topk(k, dim=-1).values


def stable_topk_ids(logits, k):
    """Resolve a tied top-k boundary by ascending token id, as the official tail."""
    shape = logits.shape
    vocab = shape[-1]
    logits = logits.reshape(-1, vocab)
    threshold = _topk_values(logits, k)[..., -1:]
    ids = torch.arange(vocab, dtype=torch.float32, device=logits.device)
    # All strict winners precede boundary ties. Every eligible key is unique,
    # so the second topk does not depend on the backend's tie ordering.
    priority = torch.where(logits > threshold, ids, ids + vocab)
    priority = torch.where(logits >= threshold, priority, ids + 2 * vocab)
    selected = -_topk_values(-priority, k)
    return selected.remainder(vocab).to(torch.int32).sort(dim=-1).values.reshape(*shape[:-1], k)


class DSpineCodeBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.dflash_config
        bridge = cfg["selected_prefix_code_bridge"]
        self.hidden_size = h = config.hidden_size
        self.code_size = c = int(bridge["code_size"])
        self.rank = r = int(bridge["rank"])
        self.eps = float(bridge["eps"])
        self.active_layers = frozenset(i - 1 for i in cfg["selected_prefix_feature_layers"])
        n = config.num_hidden_layers
        self.prev = nn.Linear(c, r, bias=False, dtype=torch.bfloat16)
        self.gate = nn.Linear(c, r, bias=True, dtype=torch.bfloat16)
        self.predecessor_gate = nn.Linear(c, r, bias=False, dtype=torch.bfloat16)
        self.out = nn.Linear(r, c, bias=False, dtype=torch.bfloat16)
        for name, shape in {
            "codebook": (config.vocab_size, c),
            "read_weight": (c, h),
            "early_read_weights": (n - 1, c, h),
            "feature_write_weights": (n, h, r),
        }.items():
            self.register_parameter(name, nn.Parameter(torch.empty(shape, dtype=torch.bfloat16), requires_grad=False))
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_parameter(name, nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False))
        self.register_buffer("packed_weight", None, persistent=False)

    def prepare(self):
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(f"Unsupported DSpine {name}")
        if self.out.weight.count_nonzero().item():
            raise ValueError("DSpine terminal-disabled projection must be zero")
        self.packed_weight = torch.cat((self.gate.weight, self.predecessor_gate.weight, self.prev.weight)).contiguous()

    def forward(self, hidden, residual, input_ids, layer, block_size):
        h, c, r = self.hidden_size, self.code_size, self.rank
        raw = (hidden.float() + residual.float()).to(torch.bfloat16) if residual is not None else hidden
        raw = raw.reshape(-1, block_size, h)
        rms = raw.float().square().mean(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        normalized = (raw.float() / (rms + self.eps)).to(torch.bfloat16)
        weight = (self.read_weight if layer == self.feature_write_weights.shape[0] - 1
                  else self.early_read_weights[layer])
        projected = F.linear(normalized, weight).float()
        codes = (math.sqrt(c) * projected / (projected.square().sum(-1, keepdim=True).sqrt() + self.eps))
        anchors = input_ids.reshape(-1, block_size)[:, 0].long()
        anchor_codes = math.sqrt(c) * F.embedding(anchors, self.codebook).float()
        codes = torch.cat((anchor_codes[:, None], codes[:, 1:]), dim=1).to(torch.bfloat16)
        packed = F.linear(codes, self.packed_weight)
        query, sender, value = packed.float().split(r, dim=-1)
        message = torch.sigmoid(query[:, 1:] + sender[:, :-1] + self.gate.bias.float()) * value[:, :-1]
        message = torch.cat((torch.zeros_like(query[:, :1]), message), dim=1).to(torch.bfloat16)
        delta = F.linear(message, self.feature_write_weights[layer]).float()
        updated = (raw.float() + rms * delta).to(torch.bfloat16)
        return (updated.reshape(-1, h), raw.reshape(-1, h), rms.reshape(-1, 1),
                packed[..., :r].reshape(-1, r))


def serial_refine(candidate_ids, raw, rms, query, anchors, lm_weight, norm_weight, bridge):
    """Return [batch, steps] greedy chain with exact token predecessor codes.

    Compute the official BF16 vocabulary factors only for candidate/anchor rows.
    Accumulate base logits and transition scores in FP32, as the Triton kernel.
    Candidate ids must be ascending so ties resolve to the smallest token id.
    """
    b, steps, k = candidate_ids.shape
    r = bridge.rank
    rows = F.embedding(candidate_ids.long(), lm_weight)
    weighted_rows = rows.float() * norm_weight.float()
    base = (weighted_rows * raw.float().unsqueeze(-2)).sum(-1)
    tg = torch.matmul(weighted_rows.to(torch.bfloat16), bridge.feature_write_weights[-1]).float()
    previous = torch.cat((anchors[:, None, None].expand(b, 1, k), candidate_ids[:, :-1]), dim=1).long()
    codes = (math.sqrt(bridge.code_size) * F.embedding(previous, bridge.codebook).float()).to(torch.bfloat16)
    tsv = F.linear(codes, bridge.packed_weight[r:]).float()
    sender, value = tsv.split(r, dim=-1)
    message = torch.sigmoid(query.float().unsqueeze(-2) + sender + bridge.gate.bias.float()) * value
    scores = base.unsqueeze(-2) + rms.float().unsqueeze(-1) * torch.matmul(message, tg.transpose(-1, -2))
    transitions = scores.argmax(-1).to(torch.int32)
    lane = torch.zeros((b, 1), dtype=torch.int32, device=raw.device)
    lanes = torch.arange(k, dtype=torch.int32, device=raw.device)[None]
    candidate_ids = candidate_ids.to(torch.int32)
    selected = []
    for step in range(steps):
        # Gaudi2's gather kernel cannot gather INT64 payloads. Selection as
        # a masked INT32 reduction also keeps the whole chain on device.
        lane = torch.where(lanes == lane, transitions[:, step], 0).sum(-1, keepdim=True, dtype=torch.int32)
        selected.append(torch.where(lanes == lane, candidate_ids[:, step], 0).sum(-1, keepdim=True, dtype=torch.int32))
    return torch.cat(selected, dim=1).to(torch.int32)
