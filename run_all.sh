#!/usr/bin/env bash
# FoveaMap - end-to-end reproduction.
#   ./run_all.sh              full run (data + train + eval + dashboard)
#   ./run_all.sh --smoke      timing probe only, no training
set -euo pipefail
cd "$(dirname "$0")"

THREADS=${THREADS:-2}
export OMP_NUM_THREADS="$THREADS"

echo "== [0/4] environment =="
python3 - <<'PY'
import importlib, subprocess, sys
for m, pkg in (("numpy","numpy"), ("scipy","scipy"), ("numba","numba"),
               ("PIL","pillow"), ("torch","torch")):
    try:
        importlib.import_module(m); print(f"  ok   {pkg}")
    except ImportError:
        print(f"  MISS {pkg} -> installing (cpu wheel)")
        if pkg == "torch":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch",
                                   "--index-url", "https://download.pytorch.org/whl/cpu"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
PY

if [[ "${1:-}" == "--smoke" ]]; then
  echo "== [smoke] timing only =="
  python3 -m src.dataset --help >/dev/null || true
  python3 -m src.train --smoke
  exit 0
fi

echo "== [1/4] synthetic dataset =="
python3 -m src.dataset

echo "== [2/4] train FoveaNet =="
python3 -m src.train --epochs 42 --iters 140 --batch 2 --ch 16

echo "== [3/4] evaluate =="
python3 -m src.eval --split test

echo "== [4/4] build dashboard =="
python3 -m src.export_dashboard

echo "done -> dashboard.html   metrics -> outputs/metrics.json"
