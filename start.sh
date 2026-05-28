#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running production nowcast..."
"${PYTHON_BIN}" -m ff5_predictor.cli nowcast --config config/nowcast/production.yaml

echo "Running release-gap nowcast backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/backtest_release_gap.yaml

echo "Running Ridge + TFT nowcast backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/daily_ridge_tft.yaml

echo "Building diagnostic nowcast dataset..."
"${PYTHON_BIN}" -m ff5_predictor.cli build-nowcast-dataset --config config/nowcast/daily_ridge.yaml

echo "Available models:"
"${PYTHON_BIN}" -m ff5_predictor.cli list-models
