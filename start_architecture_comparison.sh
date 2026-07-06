#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
MANIFEST_DIR="data/nowcasts/architecture_comparison_runs/${RUN_ID}"
MANIFEST_PATH="${MANIFEST_DIR}/manifest.tsv"
mkdir -p "${MANIFEST_DIR}"
printf "started_at_utc\tlabel\tconfig\trun_name\tlatest_output_dir\tstatus\n" > "${MANIFEST_PATH}"

DATE_ARGS=()
if [[ -n "${START_DATE:-}" ]]; then
  DATE_ARGS+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE:-}" ]]; then
  DATE_ARGS+=(--end-date "${END_DATE}")
fi

run_backtest() {
  local label="$1"
  local config_path="$2"
  local run_name
  run_name="$("${PYTHON_BIN}" - <<PY
from ff5_predictor.config import load_config
print(load_config("${config_path}")["nowcast"]["run_name"])
PY
)"
  echo "Running ${label}..."
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${config_path}" "${run_name}" "" "started" >> "${MANIFEST_PATH}"
  time "${PYTHON_BIN}" -m ff5_predictor.cli backtest-nowcast --config "${config_path}" "${DATE_ARGS[@]}"
  local latest_output_dir
  latest_output_dir="$(find "data/nowcasts/${run_name}" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${config_path}" "${run_name}" "${latest_output_dir}" "completed" >> "${MANIFEST_PATH}"
  sync
  echo "Completed ${label}; outputs: ${latest_output_dir}"
}

echo "Torch/device information:"
"${PYTHON_BIN}" -m ff5_predictor.cli torch-info --config config/research/architecture_tft.yaml
echo "Architecture comparison manifest: ${MANIFEST_PATH}"

run_backtest "linear Ridge/ElasticNet comparison" "config/research/architecture_linear.yaml"
run_backtest "one-model-per-factor ElasticNet comparison" "config/research/architecture_per_factor_elasticnet.yaml"
run_backtest "histogram gradient boosting comparison" "config/research/architecture_gradient_boosting.yaml"
run_backtest "TCN sequence comparison" "config/research/architecture_tcn.yaml"
run_backtest "FT-Transformer sequence comparison" "config/research/architecture_ft_transformer.yaml"
run_backtest "TFT sequence comparison" "config/research/architecture_tft.yaml"

echo "Architecture comparison runs complete."
