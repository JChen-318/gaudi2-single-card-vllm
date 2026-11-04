# Six-workload benchmark report

[← README](../README.md#benchmarks)

This report transcribes the user-supplied performance screenshot for `qwen3.8-27b-fp8-mtp`. It preserves the displayed values and units, and provides English tables plus machine-readable data. These results were **not rerun** as part of the documentation update.

## Test coverage and provenance

| Field | Visible in the supplied report |
|:---|:---|
| Model / service alias | `qwen3.8-27b-fp8-mtp` |
| Interface | OpenAI-compatible API |
| Workloads | General / Essay, Code Decode, Structured JSON, Prose & QA, Technical Doc QA, Math & Logic |
| Prompt lengths (tokens) | 512; 1,024; 2,048; 4,096; 8,192; 16,384; 32,768; 65,536; 131,072 |
| Concurrency | 1; every row shows 1/1 successful requests |
| Coverage | 6/6 workloads completed; 54 displayed measurements |
| Run date, harness/version, and software revision | Not shown |
| Sampling, output length, warm-up/cache state, and draft checkpoint | Not shown |
| Driver, SDK, OS, and hardware telemetry | Not shown in the screenshot |

The documented serving configuration pairs a **Qwen3.8-27B FP8 target** with a separate **DFlash2 FP16 draft model**; target-side MTP layers are not required. The `-mtp` suffix in the recorded service alias is a name, not a checkpoint requirement. The repository targets a single Intel Gaudi2 HL-225 (96 GiB HBM), Ubuntu 24.04, Python 3.12, and SDK 1.24.1. These are the documented project configuration and environment; the screenshot alone does not establish exact checkpoint revisions or independently verify those environment details. Do not merge these figures with the earlier version-specific experiments in the [optimization log](OPTIMIZATION.md).

**Evidence:** [Original report image](benchmarks/source-report.png) · [Detailed data](benchmarks/workload-details.csv) · [Summary data](benchmarks/workload-summary.csv)

## Workload summary

The following values come from the report's comparison table. Its “single-stream decode” column equals the steady decode column at concurrency 1, so that duplicate is omitted here. The partially clipped total-token column is not transcribed.

| Workload | Steady decode (tokens/s) | vs. General | E2E (tokens/s) | Reported TTFT P50* (ms) | Reported ITL P50* (ms) |
|:---|---:|---:|---:|---:|---:|
| General / Essay | 121.78 | Baseline | 51.98 | 8,339.7 | 28.6 |
| Code Decode | 129.98 | +6.7% | 47.55 | 8,327.2 | 29.9 |
| Structured JSON | 121.39 | -0.3% | 48.45 | 8,332.9 | 29.7 |
| Prose & QA | 123.99 | +1.8% | 49.94 | 9,583.5 | 29.4 |
| Technical Doc QA | 145.61 | +19.6% | 53.02 | 9,370.4 | 30.6 |
| Math & Logic | 112.96 | -7.2% | 43.52 | 8,372.4 | 29.3 |

**Metric note.** The screenshot labels the summary latency columns “P50.” However, each displayed TTFT summary is close to the **arithmetic mean of its nine prompt-length TTFT rows** (within 2.8 ms), rather than their median. The displayed values do not exactly reproduce those means. The summary ITL figures agree, to rounding, with the means of the per-row mean ITLs. The underlying request-level distributions and aggregation code are unavailable. We preserve the source labels in the CSV, but treat these as **reported aggregates**, not verified request-level percentiles. The steady-decode summaries also match arithmetic means of the nine displayed decode rates within their rounding precision. The E2E formula cannot be reconstructed from the image alone.

![Steady decode and end-to-end throughput for all six workloads](assets/workload-throughput.svg)

## Context-length curves

The plots below use the displayed per-row values without smoothing. Prompt lengths are equally spaced on a **log₂ axis**: each step doubles the token count.

![Prefill throughput across nine prompt lengths for the six workloads](assets/workload-prefill.svg)

![Decode throughput across nine prompt lengths for the six workloads](assets/workload-decode.svg)

## Observations supported by the displayed data

- **Highest workload-average decode:** Technical Doc QA, **145.61 tokens/s**, **+19.6%** relative to General.
- **Lowest workload-average decode:** Math & Logic, **112.96 tokens/s**, **−7.2%** relative to General. The highest-to-lowest ratio is approximately **1.29×**.
- **Highest individual decode measurement:** Technical Doc QA at **512 prompt tokens**, **220.82 tokens/s**. This is a single row, not the workload average.
- **Peak prefill:** Prose & QA at **16,384 prompt tokens**, **4,674.88 tokens/s**. Every workload's displayed prefill maximum occurs at 16,384 tokens.
- **Longest tested prompt in this report:** **131,072 tokens**. At this length, prefill ranges from **3,014.63 to 3,158.69 tokens/s**, decode from **67.42 to 144.89 tokens/s**, and TTFT from **41,511.56 to 48,367.29 ms**.

The screenshot alone does not establish why the workloads differ. It does not support conclusions about memory pressure, accuracy, draft acceptance, or run-to-run stability.

## Metric definitions and interpretation

| Metric | Interpretation in this transcription |
|:---|:---|
| Prompt tokens | Displayed input-length ladder; not independently retokenized |
| TTFT (ms) | Displayed time to first token for each prompt-length row |
| Mean ITL (ms) | Per-row column labeled average inter-token latency |
| Prefill (tokens/s) | Displayed prompt-processing throughput |
| Decode (tokens/s) | Displayed output/decoding throughput |
| Steady decode (tokens/s) | Workload-level figure reported by the comparison table |
| End-to-end / E2E (tokens/s) | Separately reported whole-request throughput; exact formula unavailable |
| vs. General | Relative change in the reported steady decode figure, rounded to one decimal |

Keep decode and E2E throughput separate. Do not infer decode throughput by taking `1000 / mean ITL`: speculative decoding and streamed chunks can produce multiple tokens per event, and the harness's definitions are unavailable. “Structured JSON” identifies a workload; the current speculative API rejects `response_format` and guided decoding. A 1/1 success status does not verify schema compliance or answer correctness.

## Complete per-prompt measurements

All 54 rows show **1/1 successful requests** at concurrency **1**. The common status is stated once here rather than repeated in every row.

<details>
<summary><strong>General / Essay — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 253.11 | 27.21 | 2,040.73 | 195.91 |
| 1,024 | 365.93 | 27.69 | 2,816.99 | 200.90 |
| 2,048 | 612.86 | 27.32 | 3,355.97 | 85.18 |
| 4,096 | 1,168.96 | 27.51 | 3,512.06 | 81.61 |
| 8,192 | 1,901.16 | 28.34 | 4,315.14 | 122.04 |
| 16,384 | 3,570.63 | 28.29 | 4,592.09 | 71.81 |
| 32,768 | 8,134.83 | 28.90 | 4,029.49 | 105.42 |
| 65,536 | 17,521.82 | 29.75 | 3,740.85 | 126.51 |
| 131,072 | 41,552.05 | 32.42 | 3,154.62 | 106.68 |

</details>

<details>
<summary><strong>Code Decode — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 278.06 | 27.80 | 2,004.95 | 131.50 |
| 1,024 | 375.54 | 37.94 | 2,855.59 | 116.31 |
| 2,048 | 714.54 | 28.11 | 2,934.73 | 133.86 |
| 4,096 | 1,107.60 | 27.68 | 3,744.64 | 140.05 |
| 8,192 | 1,860.16 | 27.77 | 4,432.75 | 143.99 |
| 16,384 | 3,566.42 | 29.07 | 4,609.17 | 125.77 |
| 32,768 | 8,075.99 | 29.25 | 4,064.01 | 121.52 |
| 65,536 | 17,479.86 | 30.08 | 3,752.21 | 111.96 |
| 131,072 | 41,511.56 | 31.54 | 3,158.69 | 144.89 |

</details>

<details>
<summary><strong>Structured JSON — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 280.92 | 28.50 | 1,988.52 | 179.56 |
| 1,024 | 377.63 | 36.93 | 2,842.36 | 123.74 |
| 2,048 | 778.14 | 27.81 | 2,695.29 | 121.07 |
| 4,096 | 1,101.49 | 27.52 | 3,766.19 | 119.23 |
| 8,192 | 1,866.81 | 27.98 | 4,417.31 | 72.59 |
| 16,384 | 3,526.84 | 28.99 | 4,661.11 | 126.10 |
| 32,768 | 8,077.41 | 28.69 | 4,063.36 | 120.54 |
| 65,536 | 17,465.75 | 29.42 | 3,755.27 | 117.54 |
| 131,072 | 41,545.84 | 31.71 | 3,156.10 | 112.10 |

</details>

<details>
<summary><strong>Prose & QA — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 281.86 | 27.89 | 2,117.03 | 163.78 |
| 1,024 | 375.40 | 28.31 | 3,109.51 | 145.82 |
| 2,048 | 773.02 | 28.87 | 3,034.24 | 164.12 |
| 4,096 | 1,155.86 | 28.46 | 4,026.41 | 115.31 |
| 8,192 | 2,058.76 | 29.13 | 4,503.32 | 112.65 |
| 16,384 | 3,936.51 | 29.47 | 4,674.88 | 160.80 |
| 32,768 | 9,184.12 | 29.42 | 4,002.60 | 77.68 |
| 65,536 | 20,138.26 | 29.52 | 3,667.28 | 108.37 |
| 131,072 | 48,367.29 | 33.30 | 3,051.22 | 67.42 |

</details>

<details>
<summary><strong>Technical Doc QA — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 267.88 | 27.58 | 2,233.49 | 220.82 |
| 1,024 | 378.18 | 27.34 | 3,035.62 | 155.99 |
| 2,048 | 779.55 | 34.60 | 2,890.63 | 90.22 |
| 4,096 | 1,117.53 | 28.35 | 3,996.67 | 167.12 |
| 8,192 | 1,926.00 | 28.06 | 4,611.34 | 119.96 |
| 16,384 | 3,846.44 | 28.76 | 4,619.25 | 171.05 |
| 32,768 | 9,041.71 | 29.04 | 3,926.57 | 146.83 |
| 65,536 | 19,764.73 | 29.97 | 3,592.67 | 133.40 |
| 131,072 | 47,235.53 | 41.99 | 3,014.63 | 105.10 |

</details>

<details>
<summary><strong>Math & Logic — 9 prompt lengths</strong></summary>

| Prompt tokens | TTFT (ms) | Mean ITL (ms) | Prefill (tokens/s) | Decode (tokens/s) |
|---:|---:|---:|---:|---:|
| 512 | 277.30 | 28.04 | 2,003.62 | 147.21 |
| 1,024 | 377.34 | 35.03 | 2,839.20 | 114.14 |
| 2,048 | 774.25 | 27.28 | 2,705.07 | 65.16 |
| 4,096 | 1,112.36 | 27.63 | 3,726.82 | 132.33 |
| 8,192 | 1,883.93 | 27.61 | 4,376.23 | 67.16 |
| 16,384 | 3,565.75 | 28.68 | 4,609.49 | 135.18 |
| 32,768 | 8,135.40 | 28.70 | 4,034.19 | 108.75 |
| 65,536 | 17,595.59 | 29.43 | 3,727.46 | 140.26 |
| 131,072 | 41,654.32 | 31.64 | 3,147.82 | 106.43 |

</details>

## Updating the visuals

The CSV files are the editable source for the workload charts. After updating them, regenerate the static SVGs with Python's standard library:

```bash
python docs/assets/make_workload_charts.py
```

Keep the source image or equivalent raw benchmark evidence alongside any future data updates, and record the checkpoint, draft model, software revision, sampling, output length, warm-up/cache state, and aggregation definitions when they become available.
