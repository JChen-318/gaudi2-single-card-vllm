#!/usr/bin/env bash
# Install vllm-core and vllm-gaudi (with flashinfer_gaudi) into the Python that has Intel's Gaudi PyTorch.
#
#   scripts/install.sh                 # into python3.12 (or $PYTHON)
#   scripts/install.sh --venv .venv    # a new venv on top of the system Gaudi PyTorch (recommended)
#
# Both packages are installed in editable mode, so the sources in this repository are what runs.
# Nothing here installs or replaces torch / habana-torch-plugin.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3.12}
VENV=""
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0

if [ -n "$VENV" ]; then
  [ -x "$VENV/bin/python" ] || "$PY" -m venv --system-site-packages "$VENV"
  PY=$(cd "$VENV" && pwd)/bin/python
fi
"$PY" -c 'import importlib.util as u, sys; sys.exit(u.find_spec("habana_frameworks") is None)' || {
  echo "habana_frameworks is not importable from $PY."
  echo "Install Intel's PyTorch package for SDK 1.24.1 first (see docs/INSTALL.md)."; exit 1; }

PIP=("$PY" -m pip)
# compat mode puts the source directories on sys.path (a .pth file), so they win over any copy that is
# already installed in the base environment; the default editable finder would not.
EDITABLE="--config-settings editable_mode=compat"
FLAGS=(--disable-pip-version-check)
in_venv=$("$PY" -c 'import sys; print(int(sys.prefix != sys.base_prefix))')
if [ "$in_venv" = 0 ] && ls "$("$PY" -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')"/EXTERNALLY-MANAGED >/dev/null 2>&1; then
  FLAGS+=(--break-system-packages)
fi

echo "== Python dependencies (pinned, --no-deps)"
"${PIP[@]}" install "${FLAGS[@]}" --no-deps -r "$ROOT/requirements/lock-sdk-1.24.1.txt"

echo "== vllm-core (VLLM_TARGET_DEVICE=empty)"
(cd "$ROOT/vllm-core" && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.28.1rc1.dev100+gfe755c889 \
  "${PIP[@]}" install "${FLAGS[@]}" --no-build-isolation --no-deps $EDITABLE -e .)

echo "== vllm-gaudi + flashinfer_gaudi (native libraries come from scripts/build_native.sh)"
(cd "$ROOT/vllm-gaudi" && VLLM_GAUDI_BUILD_FLASHINFER=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
  "${PIP[@]}" install "${FLAGS[@]}" --no-build-isolation --no-deps $EDITABLE -e .)

echo "$PY" > "$ROOT/.python"
(cd / && "$PY" - "$ROOT" <<'PY'
import importlib.metadata as md, importlib.util as iu, os, sys
root = sys.argv[1]
for p in ("torch", "habana-torch-plugin", "vllm", "vllm_gaudi"):
    print(f"  {p:22s} {md.version(p)}")
for mod in ("vllm", "vllm_gaudi", "flashinfer_gaudi"):
    where = os.path.dirname(iu.find_spec(mod).origin)
    print(f"  {mod:22s} {where}")
    if not where.startswith(root):
        sys.exit(f"{mod} resolves outside this repository ({where}); remove the other copy or use --venv")
PY
)
echo "Installed. Python recorded in .python; next: scripts/build_native.sh"
