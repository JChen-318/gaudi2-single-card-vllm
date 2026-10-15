#!/usr/bin/env bash
# Check that this host can build and run the project. Read-only: does not open the device.
#   scripts/check_env.sh [python]
set -u
PY=${1:-${PYTHON:-python3.12}}
fail=0; warn=0
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  [warn] %s\n' "$*"; warn=$((warn + 1)); }

echo "Host"
. /etc/os-release 2>/dev/null && { [ "${VERSION_ID:-}" = "24.04" ] && ok "$PRETTY_NAME" || note "$PRETTY_NAME (measured on Ubuntu 24.04)"; }
mem=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
[ "$mem" -ge 30 ] && ok "host memory ${mem} GiB" || bad "host memory ${mem} GiB (needs about 32 GiB: weights are staged through host memory)"
if [ "$(id -u)" = 0 ] || [ "$(ulimit -l)" = unlimited ]; then ok "locked-memory limit ($(ulimit -l))"
else note "locked-memory limit $(ulimit -l) KiB: run the service as root or raise memlock to unlimited (Gaudi pins host memory)"; fi
[ -e /lib/x86_64-linux-gnu/libtcmalloc.so.4 ] && ok "tcmalloc" || bad "libtcmalloc.so.4 missing (sudo apt install libgoogle-perftools4t64)"

echo "Gaudi software"
if command -v hl-smi >/dev/null; then
  ver=$(hl-smi -v 2>/dev/null | grep -oE 'hl-[0-9.]+' | head -1)
  drv=$(hl-smi -Q driver_version -f csv,noheader 2>/dev/null | head -1)
  name=$(hl-smi -Q name -f csv,noheader 2>/dev/null | head -1)
  ok "hl-smi $ver, driver ${drv:-?}, device ${name:-?}"
  case "$drv" in 1.24.*) ;; *) note "driver ${drv:-unknown}: everything here was built and measured on 1.24.1";; esac
else
  bad "hl-smi not found: install the Intel Gaudi driver and firmware first"
fi
command -v tpc-clang >/dev/null && ok "tpc-clang $(tpc-clang --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)" \
  || bad "tpc-clang not found (apt package habanalabs-tools): needed to build the custom kernels"
[ -d /usr/include/habanalabs ] && ok "/usr/include/habanalabs" || bad "/usr/include/habanalabs missing (apt package habanalabs-graph)"
[ -e /usr/lib/habanalabs/libtpc_kernels.so ] && ok "libtpc_kernels.so" || bad "/usr/lib/habanalabs/libtpc_kernels.so missing"
for t in cmake g++ objcopy; do command -v $t >/dev/null && ok "$t" || bad "$t not found"; done

echo "Python ($PY)"
if ! command -v "$PY" >/dev/null; then bad "$PY not found"; else
  PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" - <<'PY' || fail=$((fail + 1))
import importlib.metadata as md, importlib.util as iu, sys
bad = 0
def show(ok, msg):
    global bad
    print(("  [ok]   " if ok else "  [FAIL] ") + msg); bad += not ok
show(sys.version_info[:2] == (3, 12), f"python {sys.version.split()[0]} (needs 3.12)")
for pkg, want in (("torch", "2.11"), ("habana-torch-plugin", "1.24")):
    try:
        v = md.version(pkg); show(v.startswith(want), f"{pkg} {v}" + ("" if v.startswith(want) else f" (expected {want}.x)"))
    except md.PackageNotFoundError:
        show(False, f"{pkg} not installed: install Intel's PyTorch package for SDK 1.24.1")
show(iu.find_spec("habana_frameworks") is not None, "habana_frameworks importable")
for pkg in ("vllm", "vllm_gaudi"):
    try: print(f"  [info] {pkg} {md.version(pkg)}")
    except md.PackageNotFoundError: print(f"  [info] {pkg} not installed yet (scripts/install.sh)")
sys.exit(1 if bad else 0)
PY
fi

echo
if [ $fail -eq 0 ]; then echo "OK ($warn warning(s))"; else echo "$fail problem(s), $warn warning(s)"; exit 1; fi
