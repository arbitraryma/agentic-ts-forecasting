#!/usr/bin/env bash
# Full AirPassengers nested benchmark from the YAML config.
# This is the paper setting: fast_model_config is false.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m experiments.run_benchmark \
  --config configs/airpassengers.yaml
