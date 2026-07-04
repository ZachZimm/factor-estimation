#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running fully model-implied FF5 walk-forward series..."
"${PYTHON_BIN}" -m ff5_predictor.cli model-implied-series --config config/nowcast/model_implied_series.yaml

echo "Model-implied FF5 series complete."
