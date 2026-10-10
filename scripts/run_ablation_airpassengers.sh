#!/usr/bin/env bash
# AirPassengers routing ablations on the paper config.
# Uses the same outer splits for every ablation method.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.run_ablation \
  --config configs/airpassengers.yaml
