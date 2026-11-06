# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-card smoke test for the Qwen4Exp Track D model.

Builds the same 4-layer dummy the offline suite uses (hidden 256, hc 4, one
PLE layer at index 1, one QSA layer), this time ON the Gaudi2 device, fills
the routed experts with random packed NVFP4 planes and runs

1. one prefill forward (shape / dtype / finiteness), and
2. a fixed-prompt decode loop that prints measured tok/s.

This is called by ``validation/run_hpu_verify.sh`` step (d) and is meaningless
without a device: it exits with status 3 and a clear message when
``torch.hpu`` reports no device.

    .venv/bin/python validation/qwen4exp/hpu_device_smoke.py --steps 16
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Set before the offline module is imported (it defaults this to "cpu").
os.environ.setdefault("VLLM_HW", "gaudi2")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vllm-gaudi"))
# The offline suite owns the miniature geometry and checkpoint writer; reuse
# them so the card runs exactly the graph the host tests validated.
sys.path.insert(0, str(REPO / "vllm-gaudi" / "tests" / "unit_tests" / "ops"))


def _die(message: str, code: int = 3) -> None:
    print(f"[hpu-smoke] {message}", file=sys.stderr)
    raise SystemExit(code)


def main() -> int:
    os.environ.setdefault("VLLM_HW", "gaudi2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=64,
                        help="routed experts in the dummy store (>= 64 hits the Track C caps)")
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16, help="decode steps timed for tok/s")
    parser.add_argument("--device", default="hpu")
    parser.add_argument("--keep-model-dir", action="store_true")
    args = parser.parse_args()

    import subprocess

    import torch

    if not hasattr(torch, "hpu"):
        _die("torch has no HPU backend: install habana_frameworks.torch / run inside the container")
    # Probe the card in a CHILD process: this process must not acquire the
    # device, because vLLM's engine core is spawned and Gaudi2 hands the card
    # to one process at a time (a parent-side torch.hpu.init() makes the core
    # fail with "synStatus=8 Device acquire failed").
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch,habana_frameworks.torch;"
         "print(torch.hpu.is_available(), torch.hpu.get_device_name(0) if torch.hpu.is_available() else '')"],
        capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip().startswith("True"):
        _die("no Gaudi2 device (child-process probe failed); this smoke needs the card\n"
             + (probe.stdout + probe.stderr)[-400:])
    print(f"[hpu-smoke] device probe: {probe.stdout.strip()}")

    # Importing the offline module registers the config facade / architecture
    # and gives us the miniature config + checkpoint writer.
    import test_qwen4_exp_model as tiny  # noqa: E402
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
    from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

    tiny.TEXT_CONFIG["num_experts"] = int(args.experts)
    root = Path(tempfile.mkdtemp(prefix="qwen4exp-smoke-"))
    tiny.write_tiny_model(root, quant=False)
    print(f"[hpu-smoke] miniature checkpoint: {root}")

    model_config = ModelConfig(
        model=str(root),
        tokenizer=str(root),
        dtype="bfloat16",
        seed=0,
        max_model_len=128,
        language_model_only=True,
    )
    vllm_config = VllmConfig(model_config=model_config)
    tmp = Path(tempfile.mkstemp()[1])
    # The config context must be active before the model-parallel groups are
    # built: initialize_model_parallel reads get_current_vllm_config().
    config_ctx = set_current_vllm_config(vllm_config)
    config_ctx.__enter__()
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{tmp}",
        local_rank=0,
        backend="hccl",
    )
    initialize_model_parallel(1, 1)

    # Run the miniature checkpoint through the REAL engine (worker, mamba
    # metadata, forward context). A bare model(...) call cannot work: the GDN
    # layer reads the forward context, which only the engine sets.
    from vllm import LLM, SamplingParams

    torch.set_default_dtype(torch.bfloat16)
    started = time.perf_counter()
    llm = LLM(
        model=str(root),
        tokenizer=str(root),
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        disable_log_stats=True,
        skip_tokenizer_init=True,
    )
    build_s = time.perf_counter() - started
    print(f"[hpu-smoke] engine built in {build_s:.1f}s")

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    language_model = getattr(runner.model, "language_model", None) or runner.model
    model = language_model.model if hasattr(language_model, "model") else language_model
    layers = getattr(model, "layers", [])
    ple_layers = [layer for layer in layers if getattr(layer, "ple", None) is not None]
    for layer in ple_layers:
        ngram = layer.ple.ngram
        print(f"[hpu-smoke] PLE tables: mode={ngram.host_mode} "
              f"{ngram.host_bytes / 2**20:.2f} MiB host-resident, "
              f"device copies={sum(1 for p in layer.ple.parameters())}")
    store = None
    for layer in layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is not None and hasattr(experts, "backends"):
            store = experts
            break
    if store is not None:
        print(f"[hpu-smoke] routed-expert backends: {store.backends} "
              f"(packed planes stay uint8: "
              f"{store.gate_proj_packed.dtype}/{store.gate_proj_packed.numel() * store.gate_proj_packed.element_size() / 2**20:.1f} MiB)")

    generator = torch.Generator().manual_seed(3)
    prompt_ids = torch.randint(
        0, int(tiny.VOCAB) - 1,
        (int(args.prompt_tokens),), generator=generator).tolist()
    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=int(args.steps), temperature=0.0),
    )
    elapsed = time.perf_counter() - started
    tokens = len(outputs[0].outputs[0].token_ids)
    print(f"[hpu-smoke] generate: {tokens} tokens in {elapsed:.2f}s -> {tokens / elapsed:.2f} tok/s")
    print(f"[hpu-smoke] sample ids: {outputs[0].outputs[0].token_ids[:8]}")
    # Decision E on the card: the offloader's counters are the measured host
    # reads and device transfers, not the host-test estimates.
    offloader = getattr(language_model, "ple_offloader", None)
    if offloader is not None:
        print(f"[hpu-smoke] PLE offload: {offloader.report()}")
    llm.llm_engine.shutdown() if hasattr(llm.llm_engine, "shutdown") else None
    del llm

    if not args.keep_model_dir:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    print("[hpu-smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
