#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running market-only nowcast..."
"${PYTHON_BIN}" -m ff5_predictor.cli nowcast --config config/nowcast/latest.yaml

echo "Running market-only release-gap backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/backtest_release_gap.yaml

echo "Building diagnostic nowcast dataset..."
"${PYTHON_BIN}" -m ff5_predictor.cli build-nowcast-dataset --config config/nowcast/diagnostic.yaml

echo "Available models:"
"${PYTHON_BIN}" -m ff5_predictor.cli list-models
