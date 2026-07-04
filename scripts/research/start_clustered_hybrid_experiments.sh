#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running clustered Ridge with 0.95 threshold..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/research/extraction_clustered_095.yaml

echo "Running clustered Ridge with 0.98 threshold..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/research/extraction_clustered_098.yaml

echo "Running hybrid clustered Ridge with raw features plus 0.98 clustered components..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/research/extraction_clustered_hybrid_098.yaml

echo "Running hybrid group-PCA Ridge with raw features plus group PCA components..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/research/extraction_group_pca_hybrid.yaml

echo "Clustered and hybrid Ridge experiments complete."
