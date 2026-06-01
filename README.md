# FF5 Predictor

Nowcast-oriented nowcaster for daily Fama-French 5-factor values using market data.

The primary workflow estimates official FF5 values for recent dates where same-day market data exists but Kenneth French has not yet released the official factors. For nowcast date `t`, market-derived features may use complete after-close market data through `t`.

The active nowcast profile is a market-only all-candidates Ridge model. Historical official FF5 values are used as supervised labels during training, but FF5/RF lag values are not used as input features in the latest configs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Nowcast

```bash
python -m ff5_predictor.cli nowcast --config config/nowcast/latest.yaml
```

Useful commands:

```bash
python -m ff5_predictor.cli backtest-nowcast --config config/nowcast/backtest_release_gap.yaml
python -m ff5_predictor.cli build-nowcast-dataset --config config/nowcast/diagnostic.yaml
python -m ff5_predictor.cli list-models
```

The full active workflow is also available as:

```bash
./start_nowcast.sh
```

Nowcast outputs are written under `data/nowcasts/<run_name>/<timestamp>/`. The latest config also updates `data/nowcasts/latest/latest/` as a convenience copy.

Historical experiment outputs under `data/experiments`, `data/predictions`, and `data/processed` are not deleted or rewritten by the nowcast workflow.

Ridge nowcasts also write attribution artifacts under:

```text
data/nowcasts/latest/<timestamp>/attribution/
```

These include coefficient tables, per-feature contribution tables, top contribution summaries, and feature-group contribution summaries.

## Feature Extraction Experiments

Leakage-safe feature extraction backtests are available for group-wise PCA, PLS, per-target PLS, clustered feature averages, and TFT with compressed group-wise PCA inputs:

```bash
./start_feature_extraction.sh
```

Additional clustered and hybrid Ridge experiments can be run with:

```bash
./start_clustered_hybrid_experiments.sh
```

The market-only ElasticNet comparison using the same setup as the strongest Ridge backtest can be run with:

```bash
./start_elasticnet_experiment.sh
```

The refined ElasticNet grid around the first winning region can be run with:

```bash
./start_elasticnet_refined_experiment.sh
```

The per-factor ElasticNet experiment trains one independent ElasticNet per FF5 factor and uses parallel release-gap cutoffs:

```bash
./start_per_factor_elasticnet_experiment.sh
```

Individual configs live under:

```text
config/nowcast/extraction_group_pca.yaml
config/nowcast/extraction_pls.yaml
config/nowcast/extraction_per_target_pls.yaml
config/nowcast/extraction_clustered.yaml
config/nowcast/extraction_tft_group_pca.yaml
config/nowcast/extraction_clustered_095.yaml
config/nowcast/extraction_clustered_098.yaml
config/nowcast/extraction_clustered_hybrid_098.yaml
config/nowcast/extraction_group_pca_hybrid.yaml
config/nowcast/elasticnet_market_only.yaml
config/nowcast/elasticnet_market_only_refined.yaml
config/nowcast/per_factor_elasticnet_market_only.yaml
```

Each extractor is fit inside the cutoff-specific training frame during release-gap backtests, then applied to later target-date rows. Extractors are not fit on hidden official values or target-date inference rows.

The current model registry includes:

- `rolling_mean`
- `ewma`
- `ridge`
- `per_target_pls_ridge`
- `elasticnet`
- `per_factor_elasticnet`
- `tft`

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

To compare model run outputs against official FF5 factors, pass a predictions CSV or a nowcast run directory. The viewer adds a series mode switch (`Official`, `Predicted`, `Both`, `Error`) plus model and gap-day filters for backtest files:

```bash
python -m ff5_predictor.cli view-factors \
  --config config/nowcast/backtest_release_gap.yaml \
  --run-dir data/nowcasts/<run_name>/latest \
  --model-type ridge \
  --gap-day 1 \
  --open-browser
```

You can also point directly at a predictions file:

```bash
python -m ff5_predictor.cli view-factors \
  --config config/default.yaml \
  --predictions-csv data/nowcasts/<run_name>/latest/predictions/release_gap_predictions.csv \
  --model-type ridge
```

## Data

Fama-French data is loaded from `getFamaFrenchFactors` when practical, otherwise from the official Kenneth French remote zip. Existing repository-local FF5 CSV or zip files are ignored. Factor values are stored internally as decimal returns, so a Kenneth French value of `1.25` becomes `0.0125`.

Market data is downloaded with `yfinance` using `auto_adjust=True`.

Downloaded and cleaned data is cached as Parquet with JSON metadata sidecars under `data/cache`.

## Outputs

- `data/nowcasts/<run_name>/<timestamp>/predictions/latest_nowcast.csv`
- `data/nowcasts/<run_name>/<timestamp>/attribution/*.csv`
- `data/nowcasts/<run_name>/<timestamp>/metrics/*.json`
- `data/nowcasts/<run_name>/<timestamp>/predictions/release_gap_predictions.csv`
- `data/nowcasts/<run_name>/<timestamp>/metrics/model_ranking.csv`

## Tests

```bash
pytest
```
