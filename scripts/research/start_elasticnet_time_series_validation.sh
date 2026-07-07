#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running ElasticNet time-series validation and coefficient-stability study..."
"${PYTHON_BIN}" -m ff5_predictor.cli validate-elasticnet --config config/research/elasticnet_time_series_validation.yaml

echo "ElasticNet validation complete."
