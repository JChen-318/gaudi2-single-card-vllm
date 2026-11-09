# QK -> PV overlap and the PV ceiling (2026-11-09 evening)

## 1. Same-engine QK/PV overlap (raw replay)

`mme_overlap_probe.py` via `raw_overlap_runs.sh` (`run_context_probe.sh scal-mme-overlap`).
Production quarter geometry: QK g-m512n9472k128 (6 descriptors, FP16 scores) and PV
g-m512n128k9472 (2 descriptors), independent SRAM buffers, 30 repeats, two passes.
QK exact, PV rel 0.072% in every case. Logs `rawov-*.log`, summary `rawov-summary.log`.

| Mode | barrier p1 / p2 | no barrier p1 / p2 |
|---|---:|---:|
| QK only | 17.36 / 17.37 | 17.09 / 17.16 |
| PV only | 15.75 / 15.86 | 15.79 / 15.72 |
| QK then PV | 24.53 / 24.40 | 23.93 / 23.83 |
| PV then QK | 24.43 / 24.45 | 23.89 / 23.87 |

T(QK) + T(PV) - T(both) = 8.6-9.1 us, about one fixed submit/wait floor (9.7-10.0 us on
small shapes). The engines run QK and PV serially; dropping the barriers saves about 0.6 us
per pair, order does not matter. QK's idle compute cannot be used by a following PV.

## 2. PV ceiling: stock compiler

`compiler_matmul_bw.py --pv [--pv-transposed]` (`cmm-pv-20261109.log`, `cmm-pvt-20261109.log`).
bmm batch 8 and 32, slope cancels launch overhead:

| Shape | slope TFLOPS | per 512Q x 9472 |
|---|---:|---:|
| [512, 9472] @ [9472, 128] | 208.8 | 5.94 us |
| [512, 37760] @ [37760, 128] | 195.2 | |
| transposed [128, 9472] @ [9472, 512] | 210.4 | 5.90 us |
| transposed [128, 37760] @ [37760, 512] | 196.5 | |

Our raw PV (512 x 9472, SRAM): about 5.8-6.8 us of work, about 210 TFLOPS. Same as the
compiler, and transposing the product (512-wide output) does not help. PV reuses each P
element only D = 128 times, so 210 TFLOPS needs about 2.05 TB/s of P+V reads; full peak would
need about 3.4 TB/s. PV is input-read-bound at about 48% of peak on this stack, not limited
by our geometry.

## Consequence for the per-block budget (128Q x 37760 keys)

| Part | Stack ceiling | Time |
|---|---|---:|
| QK, write 9.67 MB of S | 1.645 TB/s output | 5.9 us |
| PV, read 9.67 MB P + V | about 210 TFLOPS | 5.9 us |
| Serial sum (no overlap, item 1) | | about 11.8 us |
| Measured MME-only pipeline (1024Q submissions) | | 11.60 us |
| Measured real pipeline (wide-relaxed / wide) | | 12.70 / about 13 us |

The MME-only pipeline already sits at the stack floor. The earlier "about 2.8 us of MME
pipeline overhead" estimate assumed PV at 432 TFLOPS peak and is withdrawn. What remains
above MME is about 1.1-1.4 us per block (TPC/MME contention and synchronisation), about 8-10%
of attention time. FP8 P to halve PV reads failed precision before (p8_vbf16 max 0.89-2.06%,
sim-fp8pv-4layers.json).
