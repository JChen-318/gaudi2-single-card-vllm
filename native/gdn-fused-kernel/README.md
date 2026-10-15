# Fused GDN MTP kernel (v12)

A TPC kernel for the gated-delta-rule (GDN) recurrence of the verify step. One launch replaces the prepared
route's host-side slot selection, L2 normalisation, `exp(log_decay)` and the prepared core itself.

- `gdn_mtp_fused_f32_gaudi2.c`: the kernel (GUID `gdn_mtp_fused_f32_gaudi2`). Shapes as served: B ≤ 16 requests,
  T = 8 tokens (K = 7 drafts + 1), H = 16 key heads, HV = 48 value heads, K = V = 128.
- `gdn_mtp_fused_f32_gaudi2.cpp/.hpp`: glue code that registers the kernel with the TPC kernel library.
- `gdn_mtp_prepared_f32_gaudi2.c`: the rewritten prepared core (8-row tiles with a shared reduction tree). It
  replaces the kernel of the same name and is used when the fused switch is off.
- `flashinfer_gaudi/patch_fused_src.py <src_root>`: applies both kernels to a copy of
  `vllm-gaudi/csrc/flashinfer_gaudi`. It replaces the prepared core and sets its tile height to 8, then adds the
  fused kernel, its glue and the custom op `custom_op::flashinfer_gaudi_gdn_mtp_fused`. The patched tree is
  byte-identical to the source of the deployed v12 library. Build it as usual: the kernel library with CMake, the
  PyTorch ops with `pytorch/setup.py`.
- `flashinfer_gaudi/patch_fused_py.py <package_dir>`: makes the Python side of flashinfer_gaudi
  (`_native.py`, `_config.py`, `gdn_decode.py`) route the prepared MTP path through the fused op when
  `FLASHINFER_GAUDI_MTP_FUSED=1`. With the switch off, the original prepared route runs. These three files in
  `vllm-gaudi/flashinfer_gaudi/` are already patched.
- `device_gdn_test.py <label> [--fused]`: on-device check of the reference, prepared and fused routes on the same
  random inputs. It reports maximum and relative errors for the output and all eight checkpoints, plus device time.
- `sim/`: TPC-simulator harnesses (cycle counts, no device needed). `gdn_sim.cpp` runs a prepared core and
  `fused_sim.cpp` runs the fused kernel against the prepared math. Build them with `sim/build.sh`, with
  `FLASHINFER_SRC` pointing to `vllm-gaudi/csrc/flashinfer_gaudi`.

## Inputs

The recurrent state pool, the per-token checkpoint slot indices, the number of accepted tokens per request, the raw
BF16 convolution output `[q | k | v]`, the FP32 log decay and the BF16 beta. The kernel has the same outputs as the
prepared core: the FP32 attention output and all eight FP32 checkpoints.

## How it works

One index-space point handles one (value head, request) pair. For each token, the Q/K conversion, the two L2 norms,
`k·q`, `exp(log decay)` and beta are computed once and kept in local memory. The 128 state rows are then processed
in tiles of 8 rows, which stay in registers across all eight tokens. Normalisation is folded into two scalars,
`rk = 1/|k|` and `rq = scale/|q|`:

```
proj  = g·rk·(S·k)              delta = beta·(v − proj)
S'    = g·S + (delta·rk)·k      out   = g·rq·(S·q) + rq·(k·q)·(delta·rk)
```

The 16 dot products of a tile share one lane-interleaving reduction tree, so lane i of the result holds dot product
i. The tree uses `mov_dual_group`/`mov_group` lane swaps and a byte-shuffle lookup for the ±2 and ±4 lane steps.
The original core reduced each dot product separately.

## Results

| | TPC simulator, cycles (B=1) | Device per call, B=1 | Device per call, B=2 |
|---|---|---|---|
| reference route (PyTorch ops) | | 0.183 ms | 0.226 ms |
| prepared route, shipped core | 2,495,280 | 0.128 ms | 0.194 ms |
| prepared route, 8-row tiles + shared reduction tree | 1,298,056 | 0.136 ms | 0.142 ms |
| fused kernel | 1,256,454 (B=2: 2,512,908) | 0.130 ms | 0.137 ms |

Device times include each route's preparation ops. Across 4 random seeds at B=1 and B=2, the largest difference from the reference
route was 6.1e-5 in the output (reference values up to 2.3e-2) and 1.5e-7 in the state checkpoints, the same size
as the shipped prepared core's differences. In the simulator, the original core runs 2.5 M cycles; spread over 24 TPCs that
is about 63 µs per layer, which matches the device trace. The rewrite roughly halves it. In this per-call test, B=1
is dominated by host launch overhead (0.128–0.136 ms for every route), so the gain only shows at B=2. End to end, the decode
round got 0.9 ms faster (single stream 27.5 → 26.6 ms, two streams 31.6 → 30.7 ms). See docs/OPTIMIZATION.md
section 13.
