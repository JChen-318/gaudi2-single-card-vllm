# Stock compiler matmul vs our raw MME replay (2026-11-09 evening)

Question (user): is the per-MME output-side limit seen in our raw QK replay a hardware limit
or a problem in our kernel/descriptors?

Method: `compiler_matmul_bw.py` via `compiler_matmul_bw.sh` (isolated container, eager,
`PT_HPU_LAZY_MODE=0`). Plain `torch.matmul(A, B, out=C)`, BF16, C[M, 9472] in HBM,
compiler-chosen geometry and slicing. M = 8192 and 32768 (155 MB and 621 MB output); the
size slope cancels per-op launch overhead. 30 iterations queued back to back. Control:
`out.fill_` of the same bytes (HBM write ceiling). Exactness spot check passed. Log
`cmm-bw-20261109.log`.

| K | compiler output slope (both MMEs) | compiler TFLOPS slope | our raw replay (SRAM out) |
|---:|---:|---:|---|
| 32 | 1.668 TB/s | 53 | flat in K up to 256 |
| 64 | 1.638 TB/s | 105 | |
| 128 | 1.431 TB/s | 183 | 1.645 TB/s size slope (rawwall), 17.57 us at 512x9472 |
| 256 | 1.098 TB/s | 281 | |
| 512 | 0.743 TB/s | 380 (88% of 432) | compute-bound at K=512 |
| HBM fill control | 2.45 TB/s write slope | | |

Findings:
- With output-bound shapes (K = 32/64), the compiler's own matmul tops out at 1.64-1.67 TB/s
  output for both MMEs, the same as our raw replay (1.645 TB/s) and the compiler HAL cost model
  (getOutputPortsNr 8 x getSinglePortBw 200 = 1.6 TB/s).
- The destination is not the limit: the compiler writes HBM, our replay writes SRAM, and a
  plain HBM fill reaches 2.45 TB/s.
- At K = 128 our replay (1.645 TB/s) is better than the compiler's eager matmul (1.43 TB/s).
- Three independent sources agree on about 0.8 TB/s per MME output: HAL constants, stock
  compiler matmul and our replay. Our descriptors are not the cause, and no compiler-chosen
  geometry exceeds it. The external "3.2 TB/s per MME" figure is not reachable through the
  stock compiler on this stack (SynapseAI 1.24.1, image v17); its definition is still unknown.

Not covered: a hardware write-active counter; it remains possible that an undocumented
configuration beats the compiler, but the compiler's own cost model assumes the same 1.6 TB/s.
