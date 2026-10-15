#!/bin/sh
# Build a TPC-simulator harness (no device needed).
#   sh build.sh prepared <kernel.c> <outdir> [ROWS]   -> gdn_sim   (a prepared core, e.g. ../gdn_mtp_prepared_f32_gaudi2.c)
#   sh build.sh fused    <outdir>                     -> fused_sim (gdn_mtp_fused_f32_gaudi2.c)
# FLASHINFER_SRC must point to the flashinfer_gaudi C++ sources (host/ holds the prepared-core glue).
# Needs tpc-clang and the simulator libraries shipped with the Habana tools (libtpcsim_shared, libtpc_tests_core_ext).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=${HABANA_TOOLS:-/usr/lib/habanatools}
INC="-I$TOOLS/include -I/usr/include/habanalabs"
: "${FLASHINFER_SRC:?set FLASHINFER_SRC to the flashinfer_gaudi C++ source root}"
LINK="-L$TOOLS -Wl,-rpath,$TOOLS -ltpc_tests_core_ext -ltpcsim_shared"
embed() {  # <kernel.c> <name> [defines]: compile for Gaudi2 and wrap the ELF into a linkable object
  tpc-clang -Wall -O2 -march=gaudi2 $3 -I"$TOOLS/include" "$1" -c -o "$2.o"
  objcopy -I binary -O elf64-x86-64 -B i386:x86-64 "$2.o" "$2_embedded.o"
}
case "$1" in
  prepared)
    K=$2; O=$3; ROWS=${4:-8}; mkdir -p "$O"; cp "$K" "$O/gdn_mtp_prepared_f32_gaudi2.c"; cd "$O"
    embed gdn_mtp_prepared_f32_gaudi2.c gdn_mtp_prepared_f32_gaudi2 "-DMTP_PREPARED_ROWS=$ROWS"
    g++ -O2 -std=c++17 -DMTP_PREPARED_ROWS=$ROWS $INC -I"$FLASHINFER_SRC/host" "$HERE/gdn_sim.cpp" \
        "$FLASHINFER_SRC/host/gdn_mtp_prepared_f32_gaudi2.cpp" gdn_mtp_prepared_f32_gaudi2_embedded.o -o gdn_sim $LINK
    echo "built $O/gdn_sim" ;;
  fused)
    O=$2; mkdir -p "$O"; cp "$HERE/../gdn_mtp_fused_f32_gaudi2.c" "$O/"; cd "$O"
    embed gdn_mtp_fused_f32_gaudi2.c gdn_mtp_fused_f32_gaudi2
    g++ -O2 -std=c++17 $INC -I"$HERE/.." -I"$FLASHINFER_SRC/host" "$HERE/fused_sim.cpp" \
        "$HERE/../gdn_mtp_fused_f32_gaudi2.cpp" gdn_mtp_fused_f32_gaudi2_embedded.o -o fused_sim $LINK
    echo "built $O/fused_sim" ;;
  *) echo "usage: sh build.sh prepared <kernel.c> <outdir> [ROWS] | fused <outdir>"; exit 2 ;;
esac
