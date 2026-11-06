#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Phase 3 preflight for the Qwen4Exp Gaudi2 bring-up.
# Prints one [OK]/[FAIL] line per requirement group and exits with the
# number of failed groups, so a hardware session can see at a glance what
# is missing before touching the model.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
CKPT="${QWEN4EXP_CKPT:-$HOME/models/qwen4exp-flashnext-plain}"
KERNEL_SO="$REPO/vllm-gaudi/csrc/qwen4exp/build/libqwen4exp_gaudi2_kernels.so"
fails=0

ok()   { printf '[OK]    %s\n' "$1"; }
bad()  { printf '[FAIL]  %s\n' "$1"; fails=$((fails + 1)); }

echo "== Qwen4Exp Gaudi2 Phase 3 preflight =="

# 1. HPU driver
if command -v hl-smi >/dev/null 2>&1 && hl-smi >/dev/null 2>&1; then
    ok "HPU driver (hl-smi responds)"
else
    bad "HPU driver: hl-smi missing or fails (modprobe habanalabs? synStatus=8 means no device)"
fi

# 2. TPC toolchain (kernel was built here; a driver host needs it to rebuild)
if [ -x /usr/bin/tpc-clang ]; then
    ok "tpc-clang at /usr/bin/tpc-clang"
else
    bad "tpc-clang not found at /usr/bin/tpc-clang"
fi

# 3. Built TPC kernel library (guid-registered host glue)
if [ -f "$KERNEL_SO" ]; then
    ok "TPC kernel library: $KERNEL_SO"
else
    bad "TPC kernel library missing: $KERNEL_SO (run cmake in vllm-gaudi/csrc/qwen4exp)"
fi

# 4. PyTorch custom-op extension: the public hpu_custom_op_pt2.h header ships
# inside the habana-torch-plugin wheel, so no bridge source tree is needed.
# Check both the header and the built extension .so.
EXT_SO="$(ls "$REPO"/vllm-gaudi/csrc/qwen4exp/pytorch/hpu_qwen4exp_nvfp4_dequant_pt2*.so 2>/dev/null | head -1)"
if "$PY" - <<'PYEOF' >/dev/null 2>&1
from habana_frameworks.torch.utils.lib_utils import get_include_dir
from pathlib import Path
assert (Path(get_include_dir()) / "hpu_custom_op_pt2.h").is_file()
PYEOF
then
    ok "public custom-op header (hpu_custom_op_pt2.h in plugin wheel)"
else
    bad "hpu_custom_op_pt2.h not found in habana plugin include dir"
fi
if [ -n "$EXT_SO" ]; then
    ok "built custom-op extension: ${EXT_SO##*/}"
else
    bad "custom-op extension not built (cd vllm-gaudi/csrc/qwen4exp/pytorch && setup.py build_ext --inplace)"
fi

# 5. Python environment
if "$PY" - <<'PYEOF' >/dev/null 2>&1
import torch, vllm, vllm_gaudi
vllm_gaudi.register_models()
from vllm import ModelRegistry
assert "Qwen4ExpForConditionalGeneration" in ModelRegistry.get_supported_archs()
assert hasattr(torch, "hpu")
PYEOF
then
    ok "python env: torch+hpu, vllm, plugin models registered"
else
    bad "python env check failed (torch/vllm/plugin import or registration)"
fi

# 6. Converted checkpoint layout
if [ -f "$CKPT/config.json" ] && [ -f "$CKPT/model.safetensors.index.json" ]; then
    info="$("$PY" - "$CKPT" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
wm = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
shards = {v for v in wm.values()}
ple = sum(1 for n in wm if ".ple.ple_embedding." in n and "shard_" in n
          and n.endswith(".weight") and "scale" not in n)
need = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "generation_config.json"]
missing = [f for f in need if not (root / f).is_file()]
print(f"tensors={len(shards)} ple_shards={ple} missing={missing}")
PYEOF
)"
    tensors="${info%% ple*}"; ple="$(printf '%s' "$info" | sed -n 's/.*ple_shards=\([0-9]*\).*/\1/p')"
    missing="$(printf '%s' "$info" | sed -n 's/.*missing=\[\(.*\)\]/\1/p')"
    if [ "${ple:-0}" -eq 128 ] && [ -z "$missing" ]; then
        ok "checkpoint layout at $CKPT ($tensors shards, 128 PLE shards, tokenizer present)"
    else
        bad "checkpoint layout incomplete at $CKPT ($info)"
    fi
else
    bad "converted checkpoint not found at $CKPT (set QWEN4EXP_CKPT)"
fi

# 7. Host RAM for the PLE tables (pinned host gather, ~26 GiB) + HBM if up
ram_gb="$(free -g | awk '/^메모리:|^Mem:/{print $7}')"
if [ "${ram_gb:-0}" -ge 40 ]; then
    ok "free host RAM for PLE tables: ${ram_gb} GiB (need ~40 with headroom)"
else
    bad "free host RAM ${ram_gb:-?} GiB: below the ~40 GiB needed for PLE tables + trunk load"
fi
if command -v hl-smi >/dev/null 2>&1 && hl-smi --json >/dev/null 2>&1; then
    ok "HBM visible via hl-smi (96 GiB expected on one Gaudi2)"
fi

echo "== preflight done: $fails failing group(s) =="
exit "$fails"
