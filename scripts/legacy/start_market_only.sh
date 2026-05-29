#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running market-only production nowcast..."
"${PYTHON_BIN}" -m ff5_predictor.cli nowcast --config config/legacy/nowcast/market_only_production.yaml

echo "Running market-only release-gap nowcast backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_backtest_release_gap.yaml

echo "Running market-only Ridge + TFT nowcast backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_ridge_tft.yaml

echo "Building market-only diagnostic nowcast dataset..."
"${PYTHON_BIN}" -m ff5_predictor.cli build-nowcast-dataset --config config/legacy/nowcast/market_only_diagnostic.yaml

echo "Available models:"
"${PYTHON_BIN}" -m ff5_predictor.cli list-models
