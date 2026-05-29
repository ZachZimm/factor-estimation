#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Running baseline market-only release-gap backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_backtest_release_gap.yaml

echo "Running size/value ETF expansion backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_size_value.yaml

echo "Running breadth/quality/real-estate ETF expansion backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_breadth_quality_re.yaml

echo "Running global ETF expansion backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_global.yaml

echo "Running all-candidate ETF expansion backtest..."
"${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config config/legacy/nowcast/market_only_all_candidates_backtest.yaml

echo "Ticker expansion experiments complete. Inspect:"
echo "  data/nowcasts/market_only_release_gap_backtest_v1"
echo "  data/nowcasts/market_only_expansion_size_value_v1"
echo "  data/nowcasts/market_only_expansion_breadth_quality_re_v1"
echo "  data/nowcasts/market_only_expansion_global_v1"
echo "  data/nowcasts/market_only_expansion_all_candidates_v1"
