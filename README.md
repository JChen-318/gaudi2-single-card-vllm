<div align="center">

# Gaudi2 Single-Card vLLM

**27B FP8 inference on one Intel Gaudi2, with non-greedy speculative decoding.**

[![Hardware: Intel Gaudi2](https://img.shields.io/badge/Intel_Gaudi2-96_GiB_HBM-0071C5?style=flat-square)](#quick-start)
[![Gaudi SDK: 1.24.1](https://img.shields.io/badge/Gaudi_SDK-1.24.1-4B5563?style=flat-square)](#requirements)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-3FB950?style=flat-square)](LICENSE)

[Benchmarks](#benchmarks) · [Quick start](#quick-start) · [API](#api) · [Documentation](#documentation) · [Limitations](#limitations)

</div>

<p align="center">
  <img src="docs/assets/inference-path.svg" width="100%" alt="Single-card inference: DFlash2 drafts candidates, the FP8 target model verifies them, and stochastic sampling produces the output.">
</p>

A source snapshot combining vLLM, the Gaudi hardware plugin, and native kernels for **Qwen3.8-27B FP8** on a single **96 GiB Gaudi2**. It includes local DFlash2/MTP adaptations, installation and serving scripts, CPU-side validation, and performance tools.

The model name is the local checkpoint/service alias used by this project; its implementation uses the Qwen3.5 architecture. This is a community engineering snapshot, independent of Intel and the official vLLM releases.

## Benchmarks

### Six workloads, nine prompt lengths

The supplied report covers **6 workloads × 9 prompt lengths**, from **512 to 131,072 tokens**, through an OpenAI-compatible endpoint serving `qwen3.8-27b-fp8-mtp`. All **54 measurements** show **1/1 requests successful** at concurrency **1**.

<p align="center">
  <img src="docs/assets/workload-throughput.svg" width="100%" alt="Reported workload averages: steady decode ranges from 112.96 tokens/s for Math and Logic to 145.61 tokens/s for Technical Doc QA; end-to-end throughput ranges from 43.52 to 53.02 tokens/s.">
</p>

| Workload | Steady decode (tokens/s) | vs. General | End-to-end (tokens/s) | Reported TTFT* (ms) |
|:---|---:|---:|---:|---:|
| General / Essay | 121.78 | Baseline | 51.98 | 8,339.7 |
| Code Decode | 129.98 | +6.7% | 47.55 | 8,327.2 |
| Structured JSON | 121.39 | −0.3% | 48.45 | 8,332.9 |
| Prose & QA | 123.99 | +1.8% | 49.94 | 9,583.5 |
| **Technical Doc QA** | **145.61** | **+19.6%** | **53.02** | 9,370.4 |
| Math & Logic | 112.96 | −7.2% | 43.52 | 8,372.4 |

Technical Doc QA has the highest reported workload-average decode throughput, **1.29×** the lowest workload average. Peak per-row prefill is **4,674.88 tokens/s**, at a 16,384-token prompt in Prose & QA.

*Source: the [supplied screenshot](docs/benchmarks/source-report.png), transcribed without rerunning the model. The summary labels TTFT as “P50,” but its aggregation method is unavailable. The figures are close to the means across the nine prompt-length rows; treat them as reported aggregates, not verified request-level percentiles. The screenshot does not establish sampling settings, output lengths, draft-model identity, or software revision. “Structured JSON” is the report's workload label; it does not establish constrained-decoding support.*

→ **[Full results, context-length curves, and metric notes](docs/BENCHMARKS.md)** · [54-row data](docs/benchmarks/workload-details.csv) · [Summary data](docs/benchmarks/workload-summary.csv)

<details>
<summary><strong>Earlier engineering measurements — different test protocol</strong></summary>

The original repository reports the following on one Gaudi2 with a 27B FP8 target and DFlash2 draft model. These runs use `temperature=1.0`, `top_p=0.95`, `top_k=20`, thinking disabled, and 512 output tokens. Short-prompt results are medians of three runs from v14; long-context decode is from v17 and TTFT from v16. They are separate from the supplied six-workload report above.

| Scenario | Streaming decode | TTFT |
|:---|---:|---:|
| General explanations and essays, one stream | 80–116 tokens/s | — |
| Python / Go / TypeScript, one stream | 164–196 tokens/s | — |
| JSON extraction and long repetition, one stream | 273–294 tokens/s | — |
| Chinese prose, one stream | 53 tokens/s | — |
| Code and reasoning, two concurrent streams combined | 243–312 tokens/s | — |
| ~55K-token context, one stream | 96 tokens/s | 14.2 s |
| ~95K-token context, one stream | 85 tokens/s | 26.7 s |
| ~180K-token context, one stream | 82 tokens/s | 59.7 s |
| Prefix-cache hit at ~57K / ~95K tokens | — | 0.9 s |

The configured per-request context limit is **262,144 tokens**, including prompt and output. This is a configuration limit, not a fully validated benchmark claim. See the [optimization log](docs/OPTIMIZATION.md) for version-specific measurements and conditions.

</details>

## What is optimized

- **Stochastic speculative decoding.** Local DFlash2/MTP paths retain temperature, top-p, and top-k sampling. Gains depend on draft acceptance and workload content.
- **Long-context attention.** Verification tokens share a historical KV prefix; dense attention reads the context together. A contiguous KV mirror avoids gathering the entire cache every decode round for a single long request.
- **Prefill and prefix reuse.** Chunked prefill uses finer context buckets and batched attention slices. Prefix caching combines vLLM attention KV reuse with device-side GDN state snapshots.
- **Host/device overlap.** Lookahead submission, device-side draft preparation, vectorized indices, and compiled regions reduce host preparation and submission gaps.
- **Native kernels and FP8.** Fused GDN recurrence, specialized rejection sampling, faster candidate selection, FP8 vocabulary projection, and FP8 KV storage target device time and memory use.
- **Image input and recipe caching.** Fixed vision buckets compile during warm-up. A TPC kernel-library shim enables device recipe caching across the native libraries.

Implementation details and before/after measurements are in the [optimization log](docs/OPTIMIZATION.md).

## Quick start

### Requirements

| Component | Tested configuration |
|:---|:---|
| Accelerator | **1× Intel Gaudi2 HL-225**, 96 GiB HBM |
| Host | x86-64, approximately 32 GiB+ RAM; approximately 31 GiB for model files |
| OS / Python | Ubuntu 24.04 / Python 3.12 |
| Gaudi software | **SDK 1.24.1**, matching driver/firmware, `hl-smi`, `tpc-clang`, and graph headers |
| PyTorch | Intel's SDK 1.24.1 build: `torch 2.11.0a0`, `habana-torch-plugin 1.24.1.482` |
| Build tools | `cmake`, `g++`, `objcopy`, `libtcmalloc.so.4` |
| Checkpoints | Local Qwen3.8-27B FP8 weights with MTP layers and the matching DFlash2 draft model |

Install the matching Gaudi driver and Intel PyTorch packages first; see the [installation guide](docs/INSTALL.md). Model weights and compiled libraries are not included.

### Install and serve

Run these commands on the Gaudi host:

```bash
git clone https://github.com/JChen-318/gaudi2-single-card-vllm.git
cd gaudi2-single-card-vllm

scripts/check_env.sh                 # Check the driver, compiler, Python, and PyTorch
scripts/install.sh --venv .venv      # Install the bundled core and plugin in editable mode
scripts/build_native.sh             # Build the native kernels and TPC shim

cp config/serve.env.example config/serve.env
vi config/serve.env                  # Set TARGET_MODEL and DRAFT_MODEL

scripts/serve.sh --daemon
scripts/smoke_test.sh --wait
```

**Allow 30–45 minutes for the first start**, including model loading and graph compilation. Later starts take approximately 25 minutes with cached device recipes. `scripts/serve.sh --skip-warmup` can accept requests in about 6 minutes for a functional check; new shapes still compile on first use, so benchmark after full warm-up.

Logs default to `~/.cache/gaudi2-single-card-vllm/logs/`. Stop the service with `scripts/stop.sh`. Non-root users need an unlimited memlock allowance; the tested root setup uses `sudo scripts/serve.sh --daemon` with an explicit `DATA_DIR` in `serve.env`.

<details>
<summary><strong>Docker deployment</strong></summary>

The host needs the matching **1.24.1 driver/firmware**, Docker 24+, and the Compose plugin. The image builds on Intel's SDK 1.24.1 PyTorch image and compiles the native libraries inside the container.

```bash
cp config/serve.env.example config/serve.env
vi config/serve.env                  # Set models, DATA_DIR, and container bind address
cp docker/docker.env.example docker/docker.env
vi docker/docker.env                # Set SERVE_ENV, MODEL_ROOT, DATA_DIR, and device paths

docker compose -f docker/compose.yaml --env-file docker/docker.env up -d --build
scripts/smoke_test.sh --wait
```

Keep the host and container paths consistent. The container uses host networking and `restart: unless-stopped`. See the [Docker guide](docs/DOCKER.md) for device mappings, startup, and updates.

</details>

## API

The OpenAI-compatible endpoint listens on **`http://127.0.0.1:8000/v1`** by default.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-fp8-mtp",
    "messages": [{"role": "user", "content": "Write a quicksort implementation in Python."}],
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

With the OpenAI Python client installed (`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="qwen3.8-27b-fp8-mtp",
    messages=[{"role": "user", "content": "Explain speculative decoding."}],
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(response.choices[0].message.content)
```

Defaults are `temperature=1.0`, `top_p=0.95`, and `top_k=20`; requests can override them. Thinking is controlled by `chat_template_kwargs.enable_thinking`.

| Capability | Current behavior |
|:---|:---|
| Streaming | Supported |
| Image input | OpenAI `image_url` parts, URL or base64; up to 4 images/request, resized to approximately 1 megapixel each; disable with `VISION=0` |
| Tool calling | `tool_choice="auto"`; forced tool selection is rejected |
| Structured-output API | `response_format` / guided decoding are rejected on the speculative path |
| Other rejected options | Penalties, `min_p`, `min_tokens`, `logit_bias`, `logprobs`, and thinking budgets |
| Video / audio | Unsupported |

For remote access, set `API_KEY`, bind `HOST=0.0.0.0`, and use a firewall or reverse proxy.

## Configuration and validation

Edit **[`config/serve.env`](config/serve.env.example)** for model paths, the endpoint, context length, concurrency, cache size, and CPU affinity. Defaults include `MAX_MODEL_LEN=262144`, `MAX_NUM_SEQS=2`, and a **29 GiB FP8 KV cache**. The decode KV mirror uses an additional 8 GiB; prefix-state snapshots also consume device memory.

`DATA_DIR` holds logs, runtime controls, and compile caches. Cache paths include native-library hashes, so rebuilding kernels selects a fresh cache. After source updates, rerun installation if dependencies changed, rebuild native libraries if their sources changed, and restart. See the [configuration guide](docs/CONFIG.md) for tuning flags and systemd setup.

```bash
scripts/bench.sh          # Single-stream, two-stream, and ~30K-context checks
scripts/bench.sh --full   # Also run the 15-category prompt suite
```

These scripts are the repository's own checks; they do not reproduce the supplied screenshot's external six-workload protocol. See [CPU-side validation](validation/README.md) and [benchmark/trace tools](tools/README.md) for more diagnostics.

## Documentation

The README and the new benchmark report are in English. Existing detailed engineering guides are in Chinese.

| Guide | Contents |
|:---|:---|
| [Benchmark report](docs/BENCHMARKS.md) | All 54 measurements, charts, source image, and metric definitions |
| [Installation](docs/INSTALL.md) | Gaudi driver, Intel PyTorch, dependencies, and troubleshooting |
| [Configuration](docs/CONFIG.md) | Service settings, optimization switches, concurrency, and systemd |
| [Optimization log](docs/OPTIMIZATION.md) | Bottleneck analysis, implementation changes, and historical results |
| [Docker](docs/DOCKER.md) | Building, deployment, restart behavior, and operations |
| [Validation](validation/README.md) / [Tools](tools/README.md) | Offline checks, benchmarking, and trace analysis |
| [Source provenance](docs/SNAPSHOT.md) | Component origins and snapshot verification |

<details>
<summary><strong>Repository map</strong></summary>

```text
scripts/                  Install, build, serve, smoke-test, and benchmark entry points
config/                   Service configuration template
docker/                   Dockerfile, Compose configuration, and entrypoint
requirements/             Dependency versions for the tested SDK
vllm-core/                Bundled vLLM core source
vllm-gaudi/               Gaudi plugin and FlashInfer-Gaudi Python/C++/TPC sources
native/mtp-sampling/      Stochastic rejection sampling kernels and PyTorch operator
native/tpc-kernel-shim/   Multi-library shim for device recipe caching
native/gdn-fused-kernel/  Fused GDN kernel, simulator, and device verification
validation/               CPU-side validation
tools/                    Benchmarks and trace analysis
examples/                 Historical standalone launchers
docs/                     Guides, benchmark data, and source provenance
```

</details>

## Limitations

- **Scope.** Tuned for one Gaudi2 and one or two concurrent requests. Higher concurrency and other model/draft combinations need independent validation.
- **Long-context latency.** Uncached prompts can take tens of seconds to reach the first token. In the supplied report, TTFT at 131,072 tokens ranges from **41.51 to 48.37 seconds**. Long prefill can also slow another request's decode to a few tokens/s.
- **Occasional recompilation.** Simultaneous short requests can trigger 90–100-second Dynamo recompilations. Startup priming and recompilation logs help diagnose this unresolved issue.
- **Prefix-cache boundaries.** GDN snapshots are saved every 16,384 tokens and at the last 2,048-token boundary before each prompt ends; at most 16 snapshots are retained. Changed conversation history can reduce reuse.
- **Quality evaluation.** FP8 KV storage is uncalibrated and lacks a systematic quality evaluation. The historical FP8 vocabulary-projection check reports 98.4% top-1 agreement with BF16.
- **Workload dependence.** Draft acceptance varies with content; prose can gain less than code or repetitive structured text. A successful benchmark request establishes execution, not answer quality or constrained-output correctness.
- **Acceptance coverage.** Tool calling, cancellation/preemption, and contexts above 100K tokens need broader validation. The supplied 131,072-token measurements and earlier ~180K runs do not establish full 262,144-token coverage.

## Provenance and license

Built on the 1Cat-vLLM-Gaudi engineering branch, the [vLLM Gaudi plugin](https://github.com/vllm-project/vllm-gaudi), and [vLLM](https://github.com/vllm-project/vllm). Existing FP8, normalization, positional encoding, and some GDN functionality are inherited; this snapshot adds single-card adaptations and optimizations.

The main project uses [Apache-2.0](LICENSE); third-party components retain their own licenses. Some adaptations, tests, and documentation used AI assistance and still require human review and validation on the target environment.
