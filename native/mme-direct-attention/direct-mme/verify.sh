#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Each invocation opens the device exclusively. Do not run alongside another HPU workload.
# The two smoke programs use the verified server PCI address 0000:08:00.0.
timeout 20 build/dma_smoke
timeout 20 build/queue_smoke
for mode in hbm sram chain; do
    for seed in 1 17 20261108; do
        timeout 20 build/mme_runner --mode "$mode" --seed "$seed"
    done
    timeout 20 build/mme_runner --mode "$mode" --pattern identity --seed 42
done
printf '%s\n' 'SUITE_PASS: DMA, MME queue, 12 GEMM cases, 262144 compared BF16 elements'
