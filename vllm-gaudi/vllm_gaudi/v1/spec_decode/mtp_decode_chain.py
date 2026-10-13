# SPDX-License-Identifier: Apache-2.0
"""Fixed-depth causal draft chain. CPU preparation stays outside compilation."""
import torch


def draft_decode_chain(model, hidden, positions, metadata_steps, alive_steps):
    """Return all drafts with the same per-step KV writes and hidden-state feed.

    Compile with fullgraph=True: unsupported adapter/context operations must
    fail qualification, not silently break this into the original Python loop.
    This is not seven/four independent parallel predictions.
    """
    drafts = []
    for step in range(alive_steps.shape[0]):
        token, inputs, next_positions = model.compute_mtp_step(hidden, positions, alive_steps[step])
        drafts.append(token)
        if step + 1 < alive_steps.shape[0]:
            positions = next_positions
            hidden = model(input_ids=inputs, positions=positions, hidden_states=hidden,
                           inputs_embeds=None, attn_metadata=metadata_steps[step])
            hidden = hidden.reshape(-1, hidden.shape[-1])
    return torch.stack(drafts, 1)


class DraftDecodeChain(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden, positions, metadata_steps, alive_steps):
        return draft_decode_chain(self.model, hidden, positions, metadata_steps, alive_steps)
