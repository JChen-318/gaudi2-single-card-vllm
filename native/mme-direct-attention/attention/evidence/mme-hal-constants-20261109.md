# Gaudi2 MME library constants (2026-11-09; unit interpretation corrected)

Source: `/usr/lib/habanalabs/libSynapseMme.so` in image `gaudi2-single-card-vllm:v17`
(SynapseAI 1.24.1-482). Command: `nm -D -C` to locate `Gaudi2::MmeHalReader::*`, then
`objdump -d` of each getter (each is `mov $imm,%eax; ret`). No REA used.

| Getter | Address | Return |
|---|---|---:|
| getDcoreNr / getDieNr / getSizeOfKernelSizeDim0 / getSizeOfOuterLoopSizeMinus1 | 0x110b50 | 1 |
| getMmeNr | 0x110b60 | 2 |
| getAccumsNr / getEuNr / getSizeOfDescNumIterMinus1 | 0x110b70 | 4 |
| getClSize | 0x110b80 | 0x80 = 128 |
| getOutputPortsNr | 0x110b90 | 8 |
| getPortSizeInElements | 0x110ba0 | 0x40 = 64 |
| getSBSize | 0x110bb0 | 0xad0 = 2768 |
| getMaxSBReuse | 0x110bc0 | 0xf0 = 240 |
| getSinglePortBw | 0x110bd0 | 0xc8 = 200 |
| getMemoryClSize | 0x110c80 | calls getClSize (128) |
| getClkFreqMHz | 0x110ee0 | 0x672 = 1650 |

Correction: see `mme-bw-unit-verdict-20261109.json`. The traced Gaudi3/common consumer uses GB/s, but the Gaudi2 memory-attributes callback is empty in this build. Products of 200 and port counts are conditional arithmetic, not demonstrated Gaudi2 hardware limits.

Historical interpretation with corrections:
- 8 output ports chip-wide = 2 MMEs x (AguOut0, AguOut1) x (master, slave): all QK captures
  enable all four per MME (fixtures/*/mme_debug, `AguOut0/1 masterEn 1 slaveEn 1`).
- 8 x 200 GB/s = 1.6 TB/s output; 4 x 200 = 0.8 TB/s per MME.
- Measured (raw_wall_runs.sh, raw_bw2_runs.sh; difference method, equal descriptor
  counts): both MMEs 1.59-1.70 TB/s, one MME 0.82 TB/s; independent of K reads
  (M1024 vs M128 equal time at equal output bytes), proportional to output bytes
  (FP32 vs FP16), same for SRAM or HBM destination.
- 200 GB/s / 1.65 GHz is about121 B/cycle numerically; proximity to a128-B cache line does not prove physical bus width or issue rate.
- WITHDRAWN: a hypothetical 200 bytes/cycle gives chip output ~2.64 TB/s at 1.65 GHz, but measured throughput below that peak does NOT contradict it. Measurement alone cannot determine the getter unit.
- Caveat (added 2026-11-09 evening): the measured slopes are CPU steady_clock intervals
  (own_submit + host wait), not write-active cycles. All QK shapes used K=128, so FLOPs scale
  exactly with output elements: the slope cannot separate compute from writes. 1.645 TB/s FP16
  at K=128 = 210 TFLOPS, about 49% of the 2x256x256 MAC x 1.65 GHz BF16 peak. Full-rate
  compute at K=128 would emit about 3.4 TB/s (both MMEs). The FP16 vs FP32 output pair (same
  FLOPs) is the only output-side-specific evidence; a constant-output K sweep and profiler
  write-active counters are still needed. Do not cite 1.6 TB/s as a proven physical port limit.
- The 11.60 us/128Q MME-only profile figure is a pipeline average (QK+PV+submit/wait+overlapped
  DMA+TPC normalisation) and must not be used to derive write bandwidth.
- External single-MME output3.2TB/s remains unverified, not disproved. Later CPU geometry queries established A/B/C port counts4/4/4 per logical MME and8/8/8 chip for the16 tested BF16 configurations.
