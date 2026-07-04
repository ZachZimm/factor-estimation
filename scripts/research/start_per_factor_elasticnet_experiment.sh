#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running market-only per-factor ElasticNet backtest with parallel cutoffs..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/research/per_factor_elasticnet_market_only.yaml

echo "Per-factor ElasticNet experiment complete."
