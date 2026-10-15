"""Route the prepared MTP core through the fused kernel when FLASHINFER_GAUDI_MTP_FUSED=1.
python3 patch_fused_py.py <flashinfer_gaudi package dir>"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(rel, pairs):
    path = root / rel
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:80])
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8")


edit("_native.py", [(
    'def public_mtp_prepared_op() -> Callable | None:\n'
    '    return _resolve_op("custom_op", "flashinfer_gaudi_gdn_mtp_prepared")\n',
    'def public_mtp_prepared_op() -> Callable | None:\n'
    '    return _resolve_op("custom_op", "flashinfer_gaudi_gdn_mtp_prepared")\n\n\n'
    'def public_mtp_fused_op() -> Callable | None:\n'
    '    return _resolve_op("custom_op", "flashinfer_gaudi_gdn_mtp_fused")\n',
)])

edit("_config.py", [(
    'def mtp_prepared_enabled() -> bool:\n',
    'def mtp_fused_enabled() -> bool:\n'
    '    """Use the fused prepared core: state slot, Q/K norms, exp(decay) and BF16 inputs in-kernel."""\n'
    '    return os.environ.get("FLASHINFER_GAUDI_MTP_FUSED", "0").strip().lower() in ("1", "true", "yes", "on")\n\n\n'
    'def mtp_prepared_enabled() -> bool:\n',
)])

edit("gdn_decode.py", [
    ("from flashinfer_gaudi._config import get_backend_policy, mtp_prepared_enabled\n",
     "from flashinfer_gaudi._config import get_backend_policy, mtp_fused_enabled, mtp_prepared_enabled\n"),
    ("    public_mtp_gdn_op,\n    public_mtp_prepared_op,\n)",
     "    public_mtp_fused_op,\n    public_mtp_gdn_op,\n    public_mtp_prepared_op,\n)"),
    ("def gated_delta_rule_mtp_packed(\n",
     "def _gated_delta_rule_mtp_fused(\n"
     "    packed_qkv: torch.Tensor,\n"
     "    log_decay: torch.Tensor,\n"
     "    beta: torch.Tensor,\n"
     "    state_pool: torch.Tensor,\n"
     "    state_indices: torch.Tensor,\n"
     "    num_accepted_tokens: torch.Tensor,\n"
     "    assume_distinct_checkpoints: bool = False,\n"
     "    checkpoint_start: int | None = None,\n"
     ") -> tuple[torch.Tensor, torch.Tensor]:\n"
     "    \"\"\"The prepared core with slot selection, Q/K L2 norms, Q scaling, exp(log decay)\n"
     "    and the BF16 casts done inside one kernel. Same outputs and checkpoint writes.\"\"\"\n"
     "    native = public_mtp_fused_op()\n"
     "    if native is None:\n"
     "        raise BackendUnavailableError(\"The fused MTP kernel is not loaded; rebuild the native extension.\")\n"
     "    batch, tokens, _ = packed_qkv.shape\n"
     "    if (checkpoint_start is not None and (not assume_distinct_checkpoints or checkpoint_start < 1\n"
     "                                          or checkpoint_start + batch * tokens > state_pool.shape[0])):\n"
     "        raise ValueError(\"Direct checkpoints require a valid, distinct, caller-proven contiguous destination.\")\n"
     "    output, checkpoints = native(state_pool, packed_qkv, log_decay, beta, state_indices,\n"
     "                                 num_accepted_tokens.reshape(batch))\n"
     "    if checkpoint_start is not None:\n"
     "        state_pool.narrow(0, checkpoint_start, batch * tokens).copy_(checkpoints.reshape(-1, 48, 128, 128))\n"
     "    elif assume_distinct_checkpoints:\n"
     "        state_pool.index_copy_(0, state_indices.reshape(-1).long(), checkpoints.reshape(-1, 48, 128, 128))\n"
     "    else:\n"
     "        for token in range(tokens):\n"
     "            state_pool.index_copy_(0, state_indices[:, token].long(), checkpoints[:, token])\n"
     "    return output.to(packed_qkv.dtype), state_pool\n\n\n"
     "def gated_delta_rule_mtp_packed(\n"),
    ("        return _gated_delta_rule_mtp_prepared(\n"
     "            packed_qkv,\n"
     "            log_decay,\n"
     "            beta,\n"
     "            state_pool,\n"
     "            state_indices,\n"
     "            num_accepted_tokens,\n"
     "            assume_distinct_checkpoints,\n"
     "            checkpoint_start,\n"
     "        )\n",
     "        prepared_core = (_gated_delta_rule_mtp_fused if mtp_fused_enabled() and public_mtp_fused_op() is not None\n"
     "                         else _gated_delta_rule_mtp_prepared)\n"
     "        return prepared_core(\n"
     "            packed_qkv,\n"
     "            log_decay,\n"
     "            beta,\n"
     "            state_pool,\n"
     "            state_indices,\n"
     "            num_accepted_tokens,\n"
     "            assume_distinct_checkpoints,\n"
     "            checkpoint_start,\n"
     "        )\n"),
])
print("patched", root)
