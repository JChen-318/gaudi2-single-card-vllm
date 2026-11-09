#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
T=${HLTHUNK_SOURCE:-/opt/habanalabs/src/hl-thunk}
L=${HLTHUNK_LIB:-/usr/lib/habanalabs}
mkdir -p build
flags=(-std=gnu11 -O2 -Wall -Wextra -Werror -I/usr/include/drm
       "-I$T/include" "-I$T/include/uapi" "-I$T/include/specs" "-I$T/include/specs/common")
for name in dma_smoke queue_smoke mme_runner; do
    gcc "${flags[@]}" "$name.c" "-L$L" "-Wl,-rpath,$L" -lhl-thunk-tests -lhl-thunk -lm -o "build/$name"
done
