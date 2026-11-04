"""Regenerate the README's static SVG visuals from the supplied benchmark CSVs.

Usage: python docs/assets/make_workload_charts.py
Requires only Python's standard library. See docs/BENCHMARKS.md for provenance.
"""

import csv
from html import escape
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "benchmarks"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
BG, FG, MUTED, GRID = "#0d1117", "#f0f6fc", "#a5b4c5", "#303b4a"
BLUE, GREEN = "#58a6ff", "#3fb950"
COLORS = ["#60a5fa", "#a3cf80", "#edc76e", "#df9c65", "#cd8dea", "#f17889"]


def begin(height, title, description):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
        f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
        f"  <title id=\"title\">{escape(title)}</title>",
        f"  <desc id=\"desc\">{escape(description)}</desc>",
        f'  <rect width="1200" height="{height}" rx="16" fill="{BG}"/>',
        f'  <g font-family="{FONT}">',
    ]


def text(x, y, value, size=22, color=FG, weight=400, anchor="start"):
    return (f'    <text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color=GRID, width=1, extra=""):
    return (f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" {extra}/>')


def save(name, parts):
    (HERE / name).write_text("\n".join([*parts, "  </g>", "</svg>", ""]), encoding="utf-8")


def inference_path():
    parts = begin(320, "Single-card speculative inference",
                  "The DFlash2 path drafts candidate tokens, verifies them with the FP8 target model, "
                  "and uses stochastic sampling to continue generation on a single Intel Gaudi2.")
    parts += [text(56, 48, "INTEL GAUDI2 / 96 GiB HBM", 20, BLUE, 600),
              text(1144, 48, "27B FP8", 20, MUTED, anchor="end"),
              text(56, 112, "Speculative decoding, one card.", 54, weight=650),
              text(56, 151, "DFlash2 path · shared KV prefixes · non-greedy sampling", 25, MUTED)]
    for x, label, sub in [(56, "Draft candidates", "DFlash2 draft model"),
                          (448, "Verify with target", "FP8 attention + GDN"),
                          (840, "Sample & continue", "Stochastic output")]:
        parts += [line(x, 191, x + 304, 191, BLUE, 2),
                  text(x, 230, label, 26, weight=600),
                  text(x, 267, sub, 22, MUTED)]
    for x in [394, 786]:
        parts += [line(x, 230, x + 22, 230, MUTED, 2),
                  f'    <path d="M{x+15} 223 L{x+22} 230 L{x+15} 237" '
                  f'fill="none" stroke="{MUTED}" stroke-width="2"/>']
    save("inference-path.svg", parts)


def throughput(summary):
    parts = begin(700, "Six-workload throughput comparison",
                  "Reported workload-average steady decode and end-to-end throughput at concurrency one. "
                  "Technical Doc QA leads at 145.61 tokens per second steady decode. Full data is in docs/BENCHMARKS.md.")
    parts += [text(56, 54, "Six workloads. One Gaudi2.", 38, weight=650),
              text(56, 91, "512–131,072 prompt tokens · concurrency 1 · tokens/s", 23, MUTED)]
    for x, color, label in [(56, BLUE, "Steady decode"), (340, "#64748b", "End-to-end")]:
        parts += [line(x, 124, x + 26, 124, color, 8), text(x + 40, 131, label, 22, MUTED)]
    parts.append(text(1144, 131, "vs. General", 20, MUTED, anchor="end"))
    x0, width = 344, 660
    for value in range(0, 161, 40):
        x = x0 + value / 160 * width
        parts += [line(x, 166, x, 622), text(x, 157, value, 20, MUTED, anchor="middle")]
    for i, row in enumerate(summary):
        y = 195 + i * 78
        label = row["label"].replace("General / Essay", "General / Essay")
        steady = float(row["reported_steady_decode_tokens_per_s"])
        e2e = float(row["reported_e2e_tokens_per_s"])
        delta = float(row["reported_vs_general_percent"])
        color = GREEN if row["workload"] == "doc-qa" else BLUE
        parts.append(text(56, y + 12, label, 25, weight=600))
        for value, yy, hh, cc in [(steady, y - 10, 25, color), (e2e, y + 23, 12, "#64748b")]:
            w = value / 160 * width
            parts += [f'    <rect x="{x0}" y="{yy}" width="{w:.2f}" height="{hh}" rx="3" fill="{cc}"/>',
                      text(x0 + w + 12, yy + hh - (2 if hh == 25 else -2), f"{value:.2f}",
                           22 if hh == 25 else 19, FG if hh == 25 else MUTED, 600 if hh == 25 else 400)]
        parts.append(text(1144, y + 12, "Baseline" if i == 0 else f"{delta:+.1f}%", 23,
                          GREEN if delta > 0 else MUTED, anchor="end"))
    parts += [line(56, 644, 1144, 644),
              text(56, 679, "Source: supplied report · qwen3.8-27b-fp8-mtp · not a rerun", 20, MUTED)]
    save("workload-throughput.svg", parts)


def context_chart(summary, details, field, title, maximum, ticks, name):
    parts = begin(760, title,
                  "Six workloads across 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, "
                  "and 131072 prompt tokens. The horizontal axis is logarithmic base two. "
                  "Exact measurements are available in docs/BENCHMARKS.md and the source CSV.")
    parts += [text(56, 54, title, 38, weight=650),
              text(56, 91, "tokens/s · concurrency 1 · prompt length doubles at each step", 23, MUTED)]
    for i, row in enumerate(summary):
        x, y = 56 + (i % 3) * 370, 128 + (i // 3) * 36
        parts += [line(x, y - 6, x + 25, y - 6, COLORS[i], 4), text(x + 38, y, row["label"], 22, MUTED)]
    x0, x1, top, bottom = 120, 1124, 212, 640
    for value in ticks:
        y = bottom - value / maximum * (bottom - top)
        parts += [line(x0, y, x1, y), text(x0 - 18, y + 7, f"{value:,}", 21, MUTED, anchor="end")]
    lengths = [512 * 2 ** i for i in range(9)]
    for i, n in enumerate(lengths):
        x = x0 + i / 8 * (x1 - x0)
        label = "512" if i == 0 else f"{n // 1024}K"
        parts += [line(x, top, x, bottom, GRID, 1, 'stroke-dasharray="3 7"'),
                  text(x, bottom + 35, label, 22, MUTED, anchor="middle")]
    for i, row in enumerate(summary):
        points = sorted((r for r in details if r["workload"] == row["workload"]), key=lambda r: int(r["prompt_tokens"]))
        coords = [(x0 + j / 8 * (x1 - x0), bottom - float(r[field]) / maximum * (bottom - top)) for j, r in enumerate(points)]
        path = " ".join(f"{'M' if j == 0 else 'L'} {x:.2f} {y:.2f}" for j, (x, y) in enumerate(coords))
        parts.append(f'    <path d="{path}" fill="none" stroke="{COLORS[i]}" stroke-width="3" stroke-linejoin="round"/>')
        for x, y in coords:
            parts.append(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{COLORS[i]}" stroke="{BG}" stroke-width="1.5"/>')
    parts += [text(622, 714, "Prompt tokens (log₂ scale; 1K = 1,024)", 22, MUTED, anchor="middle"),
              text(56, 746, "Source: supplied report · exact values in docs/BENCHMARKS.md", 18, MUTED)]
    save(name, parts)


def main():
    with (DATA / "workload-summary.csv").open(encoding="utf-8", newline="") as f:
        summary = list(csv.DictReader(f))
    with (DATA / "workload-details.csv").open(encoding="utf-8", newline="") as f:
        details = list(csv.DictReader(f))
    inference_path()
    throughput(summary)
    context_chart(summary, details, "prefill_tokens_per_s", "Prefill throughput by prompt length", 5000,
                  range(0, 5001, 1000), "workload-prefill.svg")
    context_chart(summary, details, "decode_tokens_per_s", "Decode throughput by prompt length", 250,
                  range(0, 251, 50), "workload-decode.svg")
    print("Generated inference path and three workload charts.")


if __name__ == "__main__":
    main()
