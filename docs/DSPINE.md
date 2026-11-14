# Qwen3.8-27B DSpine on Gaudi2

This backend loads `tencent/Qwen3.8-27B-DSpine-b16` through the existing HPU
DFlash context-KV and target-verification pipeline. The draft executes all five
decoder layers, the learned adjacent-position code write after each layer, and
the serial exact-sender last-write refinement over 16 candidates per position.
It does not load the DFlash2 convolution or candidate selector.

Draft selection uses greedy serial refinement. Non-greedy requests use the existing HPU
target sampler and rejection path for deterministic proposals. This port does
not implement SGLang's stochastic draft proposals, DSpark adaptive verification,
or ReplaySSM CUDA kernels. The target's existing GDN checkpoints handle rollback.

## Run

Download the draft weights without changing their `config.json`:

```bash
hf download tencent/Qwen3.8-27B-DSpine-b16 \
  --revision 5f54174d301dbca65d9aa4448dc47e56732d9a12 \
  --local-dir /mnt/sas/models/Qwen3.8-27B-DSpine-b16
cp config/dspine.env.example config/dspine.env
# Edit paths and address in config/dspine.env.
scripts/serve.sh --config config/dspine.env --skip-warmup
```

With the existing v19 image, the source overlay can run in its own container:

```bash
bash scripts/serve_dspine_container.sh config/dspine.env --skip-warmup
docker logs -f gaudi2-dspine
```

The container mounts source and configuration read-only and persists runtime
files in `DATA_DIR`. Stop any other process using the same Gaudi device first.
The example binds localhost on port 8016; edit `HOST` if remote access is needed.

The safetensors SHA-256 is
`ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1`.

Use `NUM_SPECULATIVE_TOKENS=15` for the trained 16-token block, or `7` for
an 8-token prefix. The draft always uses BF16 independently of the target's
FP16/BF16 activation dtype; `DTYPE=auto` preserves the existing target choice.
Both require TP=PP=DP=1, text-only serving,
prefix caching disabled, and at most 8 concurrent sequences. Other draft/target
shapes are rejected. The existing DFlash2 configuration remains the default.
The example uses `SPEC_FAST_PATHS=1` to enable supported sampling and host
shortcuts. Set it to `0` to disable the runtime shortcuts for diagnosis.

`DSPINE=1` turns on the three things a DSpine block needs beyond the DFlash2
pipeline (measured in section 25 of [the optimization notes](OPTIMIZATION.md)):

- the host lookahead and presampling that queue the next round's target
  forward before the current round returns, and the predraft that queues the
  next draft behind sampling. Both were written for DFlash2 and work for any
  block length; they only needed their `num_speculative_tokens == 7` gate
  removed.
- `VLLM_HPU_PAST_CONCAT_COPY=1`. With the draft queued behind the target's
  sampling, Synapse rejects the draft attention's concat of the gathered past
  with the current block ("unsupported node for hardware"). Writing both parts
  into one new buffer is the same tensor and lowers cleanly.
- the native T=1 rejection correction at 16 draft positions. Its TPC kernels
  read the width from their tensors; only the glue and the Python gate were
  capped at 8. Without this a sampled request falls back to the eager
  reference, which costs about 6 ms per round.

Measured warm, single-request decode (one card, `--skip-warmup`, six
workloads at temperature 0 and 1, `tools/benchmark_dspine_decode.py`):

| draft | tokens/round | ms/round | tok/s | harmonic mean |
|---|---|---|---|---|
| DFlash2, `K=7` | 4.16 | 26.2 | 158.6 | 157.7 |
| DSpine b16, `K=15`, `DSPINE=1` | 4.85 | 28.4 | 170.6 | 170.7 |

Per workload DSpine runs 12-23% faster on code and Chinese and 7-11% slower
on English prose, where its accepted length varies most between runs. Accepted
length moves 5-10% from run to run, so a single run of either draft does not
settle a difference of that size.

The native GDN verify kernel is specialized for 8-token blocks. A full
16-token verification composes two functional 8-token kernel calls: the second
starts from the first call's final state, and all 16 checkpoints are retained.
The accepted checkpoint is selected from the entire previous block before any
writeback. Other shapes keep the general checkpoint path. A higher draft
acceptance length does not by itself establish a throughput improvement;
benchmark both block lengths on the intended workload. `--skip-warmup` compiles
unseen shapes on first use; exclude those requests from performance comparisons.

The bridge returns its last-layer raw state, RMS, and query alongside the
normalized features. This keeps refinement state explicit across compiled
regions and graph replays. It never reads a mutable Python stash from a
previous forward pass.

Candidate retrieval resolves ties at the top-16 boundary by ascending token
ID before serial refinement, matching the official tail's tie convention.
Both value-threshold and token-priority selection use chunk maxima to avoid
running top-k over every vocabulary entry. Threshold values remain exact,
and unique token-priority keys preserve the smallest-ID boundary tie rule.
The DSpine stack and candidate regions use static compilation per bucket.
Packed proposal features and anchor IDs are made contiguous before entering
the candidate region; this avoids passing sliced full-block storage to the
HPU reshape lowering when the active batch changes.

## Validation

```bash
PYTHONPATH=vllm-gaudi .venv/bin/python -m pytest -q validation/test_dspine.py \
  vllm-gaudi/tests/unit_tests/test_platform.py -k 'dspine or qwen38_dflash2'
PT_HPU_LAZY_MODE=0 PYTHONPATH=vllm-gaudi .venv/bin/python tools/check_dspine_bridge.py
.venv/bin/python tools/validate_dspine_api.py --base http://127.0.0.1:8016 \
  --output /tmp/dspine-api.json
.venv/bin/python tools/validate_dspine_greedy.py --base http://127.0.0.1:8016 \
  --output /tmp/dspine-greedy.json
```

The API check exercises repeated greedy streaming, sampled Chinese output,
and simultaneous requests, and saves the complete output and Prometheus
metrics. CPU equation tests do not substitute for device and target-only
generation comparisons.

Repeated greedy serving currently has an unresolved reproducibility issue:
identical temperature-zero requests can produce different output. Device
bridge and GDN checkpoint checks pass, but this does not establish end-to-end
equivalence to target-only decoding or an aggregate accuracy threshold. This
limitation remains open while decode throughput is evaluated separately.
Use `validate_dspine_api.py --allow-repeat-variation` for a functional smoke
check that records this variation without treating it as a blocking failure;
the default validation command still fails when greedy repeats differ.

Measure warm, single-request decode throughput with fixed generated lengths:

```bash
.venv/bin/python tools/benchmark_dspine_decode.py --label dspine-b16 \
  --output /tmp/dspine-b16.json
```

This records two warmups and three measured runs for each code, Chinese and
English prompt at temperatures 0 and 1. Decode time excludes the first
token-bearing chunk and uses actual streamed token IDs, so speculative chunks
containing several tokens are counted correctly. Output equality is recorded
but does not block this benchmark. Compare otherwise identical server settings
with `NUM_SPECULATIVE_TOKENS=7` and `15`, and the existing DFlash2 draft.

The bridge check requires exclusive access to the HPU, with the serving
engine stopped. It compares CPU and HPU eager/compiled execution for both
block sizes, including a vocabulary-sized top-k boundary tie case.

## Sources

- [Tencent DSpine](https://github.com/Tencent/DSpine), revision
  `d211869ac168b756e197f7607ff37cd342b50fb2` (Apache-2.0).
- [Draft checkpoint](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16).
- The tensor equations mirror the official bridge and greedy serial-table
  kernels; candidate-specific BF16 factors avoid allocating vocabulary-wide
  refinement tables.
