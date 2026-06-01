#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running market-only ElasticNet backtest with Ridge comparison..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/elasticnet_market_only.yaml

echo "ElasticNet experiment complete."
