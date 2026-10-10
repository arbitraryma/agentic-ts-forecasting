#!/usr/bin/env bash
# Combine saved benchmark summaries under results/benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m analysis.aggregate_results \
  --results results/benchmark \
  --output results/aggregate
