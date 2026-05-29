#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running production market-only nowcast..."
"${PYTHON_BIN}" -m ff5_predictor.cli nowcast --config config/nowcast/production.yaml

echo "Running production market-only release-gap backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/backtest_release_gap.yaml

echo "Building production diagnostic nowcast dataset..."
"${PYTHON_BIN}" -m ff5_predictor.cli build-nowcast-dataset --config config/nowcast/diagnostic.yaml
