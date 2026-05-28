# FF5 Predictor

Production-oriented nowcaster for daily Fama-French 5-factor values using market data.

The primary workflow estimates official FF5 values for recent dates where same-day market data exists but Kenneth French has not yet released the official factors. For nowcast date `t`, market-derived features may use complete after-close market data through `t`; official FF5-derived features may only use values available through the latest released official factor date.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Nowcast

```bash
python -m ff5_predictor.cli nowcast --config config/nowcast/production.yaml
```

Useful commands:

```bash
python -m ff5_predictor.cli build-nowcast-dataset --config config/nowcast/daily_ridge.yaml
python -m ff5_predictor.cli backtest-nowcast --config config/nowcast/backtest_release_gap.yaml
python -m ff5_predictor.cli list-models
```

Nowcast outputs are written under `data/nowcasts/<run_name>/<timestamp>/`. The production config also updates `data/nowcasts/production_latest/latest/` as a convenience copy.

Historical experiment outputs under `data/experiments`, `data/predictions`, and `data/processed` are not deleted or rewritten by the nowcast workflow.

## Historical Forecast Experiments

The older shifted-market-data experiment commands are still available for research comparison:

```bash
python -m ff5_predictor.cli run-all --config config/default.yaml
python -m ff5_predictor.cli run-experiment --config config/experiments/tft_patchtst_daily.yaml
```

Research experiment commands:

```bash
python -m ff5_predictor.cli list-models
python -m ff5_predictor.cli run-experiment --config config/experiments/tft_patchtst_daily.yaml
python -m ff5_predictor.cli run-experiment --config config/experiments/tft_patchtst_5d.yaml
python -m ff5_predictor.cli run-experiment --config config/experiments/tft_patchtst_residual_daily.yaml
python -m ff5_predictor.cli run-experiment --config config/experiments/tft_patchtst_residual_5d.yaml
python -m ff5_predictor.cli train-model --model tft --config config/experiments/tft_patchtst_daily.yaml
```

The experiment runner supports stronger baselines, ElasticNet, optional boosting adapters, and PyTorch sequence models:

- `rolling_mean`
- `rolling_median`
- `ewma`
- `ridge`
- `elasticnet`
- `lightgbm`
- `xgboost`
- `catboost`
- `tft`

Hidden/deprecated compatibility models can be listed with:

```bash
python -m ff5_predictor.cli list-models --include-hidden
```

## Factor Viewer

Generate a standalone browser-based FF5 data viewer:

```bash
python -m ff5_predictor.cli view-factors --config config/default.yaml --open-browser
```

The viewer includes date filters, factor toggles, summary statistics, hover values, and CSV export. Start and end dates can be set when generating the file:

```bash
python -m ff5_predictor.cli view-factors \
  --config config/default.yaml \
  --start-date 2010-01-01 \
  --end-date 2025-12-31 \
  --viewer-output data/processed/ff5_factor_viewer.html
```

## Data

Fama-French data is loaded from `getFamaFrenchFactors` when practical, otherwise from the official Kenneth French remote zip. Existing repository-local FF5 CSV or zip files are ignored. Factor values are stored internally as decimal returns, so a Kenneth French value of `1.25` becomes `0.0125`.

Market data is downloaded with `yfinance` using `auto_adjust=True`.

Downloaded and cleaned data is cached as Parquet with JSON metadata sidecars under `data/cache`.

## Outputs

- `data/nowcasts/<run_name>/<timestamp>/predictions/latest_nowcast.csv`
- `data/nowcasts/<run_name>/<timestamp>/metrics/*.json`
- `data/processed/modeling_dataset.parquet`
- `data/predictions/ff5_predictions.csv`
- `data/predictions/metrics.json`
- `data/experiments/<run_name>/predictions/*.csv`
- `data/experiments/<run_name>/metrics/*.json`

## Tests

```bash
pytest
```
