#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running group-wise PCA Ridge backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/extraction_group_pca.yaml

echo "Running PLS Ridge backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/extraction_pls.yaml

echo "Running per-target PLS Ridge backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/extraction_per_target_pls.yaml

echo "Running clustered-feature Ridge backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/extraction_clustered.yaml

echo "Running TFT with group-wise PCA inputs backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/nowcast/extraction_tft_group_pca.yaml

echo "Feature extraction experiments complete."
