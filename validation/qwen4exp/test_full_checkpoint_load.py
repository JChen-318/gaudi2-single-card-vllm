# SPDX-License-Identifier: Apache-2.0
"""Full-checkpoint loader dry-run for the Qwen4Exp port (host only).

The serve-time risk this covers: vLLM's standard loader streams the checkpoint
straight into ``load_weights``, so every non-expert tensor name has to resolve
to a real parameter (packed qkv/gate_up merges included). The Track D port
replaces the old ``--load-format qwen4exp_plain`` loader, and a single stray
prefix aborts the whole load — this actually happened with ``model.visual.*``
under ``--language-model-only`` ("There is no module or parameter named
'visual'"), which the filter test below pins down.

``test_real_checkpoint_loads_into_meta_model`` builds the real 48-layer model on
the meta device and runs the production ``load_weights`` over the real index
(301,759 tensors, shapes taken from the safetensors headers). Tensor data never
moves: ``Tensor.copy_`` is stubbed to record (destination, source) shapes and
assert they agree, and the device-only synchronize hook is a no-op. It needs the
checkpoint and ~30 s, so it is gated on ``VLLM_QWEN4EXP_FULL_LOAD=1``
(validation/run_hpu_verify.sh sets it).

Run:
  VLLM_QWEN4EXP_FULL_LOAD=1 .venv/bin/python -m pytest -q \
      validation/qwen4exp/test_full_checkpoint_load.py
"""

from __future__ import annotations

import glob
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("VLLM_HW", "cpu")

import pytest  # noqa: E402
import torch  # noqa: E402

CHECKPOINT = Path(
    os.environ.get("VLLM_QWEN4EXP_MODEL",
                   os.path.expanduser("~/models/qwen4exp-flashnext-plain")))

_SAFETENSORS_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "U32": torch.uint32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _checkpoint_headers() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    headers: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
    for shard in sorted(CHECKPOINT.glob("tensors/*.safetensors")):
        with open(shard, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            headers[name] = (_SAFETENSORS_DTYPES[meta["dtype"]], tuple(meta["shape"]))
    return headers


# ---------------------------------------------------------------------------
# Fast unit test: the filters that keep foreign prefixes out of the loader
# ---------------------------------------------------------------------------
def test_text_only_filter_drops_foreign_prefixes():
    from types import SimpleNamespace

    from vllm_gaudi.models.qwen4_exp import Qwen4ExpForCausalLM

    stub = SimpleNamespace(_experts_from_source=True, _streamed_expert_tensors=0)
    names = [
        ("model.layers.0.mlp.shared_expert.down_proj.weight", "keep"),
        ("model.visual.blocks.0.attn.qkv.weight", "drop-visual"),
        ("visual.merger.linear_fc1.weight", "drop-visual"),
        ("mtp.layers.0.mlp.gate.weight", "drop-mtp"),
        ("model.language_model.layers.1.ple.ple_embedding.shard_0.weight", "drop-ple"),
        ("model.language_model.layers.0.mlp.experts.0.gate_proj.weight", "drop-expert"),
        ("lm_head.weight", "keep"),
    ]
    kept = list(Qwen4ExpForCausalLM._filter_checkpoint_tensors(
        stub, ((name, tag) for name, tag in names)))
    assert [tag for _, tag in kept] == ["keep", "keep"]
    assert stub._streamed_expert_tensors == 1


# ---------------------------------------------------------------------------
# Full dry-run (gated: needs the 162 GB checkpoint and ~30 s)
# ---------------------------------------------------------------------------
def _require_full_load() -> None:
    if os.environ.get("VLLM_QWEN4EXP_FULL_LOAD", "") != "1":
        pytest.skip("set VLLM_QWEN4EXP_FULL_LOAD=1 to run the full checkpoint dry-run")
    if not (CHECKPOINT / "model.safetensors.index.json").exists():
        pytest.skip(f"no ModelOpt checkpoint at {CHECKPOINT}")


def test_real_checkpoint_loads_into_meta_model():
    _require_full_load()
    headers = _checkpoint_headers()
    assert len(headers) > 300_000, len(headers)

    import vllm_gaudi

    vllm_gaudi.register_models()
    import vllm.platforms as platforms

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (destroy_distributed_environment,
                                  destroy_model_parallel,
                                  init_distributed_environment,
                                  initialize_model_parallel)

    model_config = ModelConfig(
        model=str(CHECKPOINT), tokenizer=str(CHECKPOINT), dtype="bfloat16",
        seed=0, max_model_len=4096)
    vllm_config = VllmConfig(model_config=model_config)
    distributed = {"group": False, "parallel": False}
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="file:///tmp/qwen4exp_full_load",
        local_rank=0, backend="gloo")
    distributed["group"] = True
    context = set_current_vllm_config(vllm_config)
    context.__enter__()
    initialize_model_parallel(1, 1)
    distributed["parallel"] = True

    saved_device = platforms.current_platform.current_device
    saved_sync = torch.hpu.synchronize
    saved_copy = torch.Tensor.copy_
    platforms.current_platform.current_device = lambda: torch.device("cpu")
    torch.hpu.synchronize = lambda *args, **kwargs: None
    mismatches: list[tuple] = []
    copies = [0]

    def recording_copy(self, src, *args, **kwargs):
        if self.device.type == "meta" or getattr(src, "device", None) is not None and \
                getattr(src, "device").type == "meta":
            copies[0] += 1
            if tuple(self.shape) != tuple(src.shape) and getattr(src, "numel", lambda: 0)() != 0:
                mismatches.append((tuple(self.shape), tuple(src.shape)))
            return self
        return saved_copy(self, src, *args, **kwargs)

    try:
        from vllm_gaudi.models.qwen4_exp import Qwen4ExpForCausalLM

        with torch.device("meta"):
            model = Qwen4ExpForCausalLM(vllm_config=vllm_config, prefix="")
        # Expert planes and the PLE table are loaded from the checkpoint source
        # (Track B / host mmap); skip their device work in this dry-run.
        model._experts_from_source = True
        model._load_experts_from_source = lambda: set()
        model._maybe_attach_ple_tables = lambda: 0
        torch.Tensor.copy_ = recording_copy
        weights = ((name, torch.empty(shape, dtype=dtype, device="meta"))
                   for name, (dtype, shape) in headers.items())
        loaded = model.load_weights(weights)
    finally:
        torch.Tensor.copy_ = saved_copy
        torch.hpu.synchronize = saved_sync
        platforms.current_platform.current_device = saved_device
        # Leave no global state behind: the other suites in the same pytest
        # session initialise their own distributed/model-parallel state, and a
        # surviving process group makes their fixtures fail at setup.
        if distributed["parallel"]:
            destroy_model_parallel()
        if distributed["group"]:
            destroy_distributed_environment()
        context.__exit__(None, None, None)

    assert not mismatches, f"destination/source shape mismatches: {mismatches[:5]}"
    assert copies[0] > 1000, copies[0]
    # The language-model parameters, not the dropped expert/vision/PLE tensors.
    assert len(loaded) > 1000, len(loaded)
    state = set(model.state_dict())
    assert "model.embed_tokens.weight" in state
    assert "model.layers.0.mlp.shared_expert.down_proj.weight" in state
    assert not [name for name in state if ".visual." in name]
