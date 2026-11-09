# QK at constant output, K swept (2026-11-09 evening)

Question: is raw QK time set by MME compute or by the output side? Earlier slopes all had
K=128, so FLOPs scaled exactly with output elements and could not separate the two.

Method: compiler captures `g-m512n9472k{32,64,128,256,512}` (6 descriptors each, 3 per MME)
and `g-m128n1024k*` (2 descriptors, floor). `mme_raw_probe.py --k K --scores-fp16 --bulk
--repeats 20`, small-integer inputs, FP16 output 512x9472 = 9.70 MB in SRAM, exact check on all
outputs (single-engine runs: exact on the ~49% the engine owns). Script `raw_ksweep_runs.sh`,
logs `rawk-*.log`, summary `rawk-summary.log`. Timing is still CPU steady_clock around submit
and host wait (median of 20).

| K | m128n1024 floor | m512n9472 both MMEs | m512n9472 MME0 only (half output) | peak-rate compute per MME |
|---:|---:|---:|---:|---:|
| 32 | 9.87 | 17.25 | 16.77 | 0.7 us |
| 64 | 9.74 | 17.50 | 16.98 | 1.4 us |
| 128 | 10.03 | 17.57 | 17.08 | 2.9 us |
| 256 | 10.02 | 17.64 | 17.16 | 5.7 us |
| 512 | 10.27 | 23.52 | 23.11 | 11.5 us |

Peak-rate compute assumes 256x256 MAC x 1.65 GHz per MME (216 TFLOPS BF16 each).

Findings:
- K 32 -> 256 (8x FLOPs): +0.39 us. At K <= 256 the time is not compute-bound. The QK we
  run (K=128) needs about 2.9 us of compute per MME but takes about 7.6 us above the floor.
- K=512 is the first point where compute shows: about 13.3 us above the floor against 11.5 us
  at peak rate, i.e. compute and output overlap, total ~ max(compute, output) + ramp.
- One MME doing half the output takes the same time as two MMEs doing all of it (-0.5 us).
  The limit is per MME, not a shared SRAM total.
- Together with FP16 vs FP32 (same descriptors, same FLOPs, 2x bytes -> about 1.8x work time)
  and the equal-descriptor size slope (time proportional to output bytes), the K-independent
  plateau is an output-side, per-MME cost: about 4.85 MB per MME in about 7.6 us including
  ramp (marginal slope about 0.82 TB/s per MME).

Still unproven: which part of the output side (write-back port, accumulator drain or
conversion, output AGU geometry such as ROI 256 width, or credits). This is not a profiler
write-active count, and it does not explain the external 3.2 TB/s per-MME figure: our
per-MME rate is about 1/4 of it, so a geometry or configuration cause remains a candidate.
The 11.60 us/128Q pipeline figure is not used here.
