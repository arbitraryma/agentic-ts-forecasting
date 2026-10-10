#!/usr/bin/env bash
# Cheap nested-evaluation check.
# Three classical experts, two outer origins, two inner windows.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.run_benchmark \
  --config configs/airpassengers.yaml \
  --outer-windows 2 \
  --inner-windows 2 \
  --fast \
  --models seasonal_naive,ets,auto_arima
