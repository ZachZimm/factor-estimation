#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running momentum-enhanced model-implied factor series..."
"${PYTHON_BIN}" -m ff5_predictor.cli model-implied-series --config config/research/momentum_enhanced_model_implied.yaml

echo "Momentum-enhanced experiment complete."
