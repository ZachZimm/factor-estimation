from __future__ import annotations

from pathlib import Path

from ff5_predictor.config import load_config


def test_active_and_legacy_configs_load() -> None:
    active = [
        "config/nowcast/production.yaml",
        "config/nowcast/backtest_release_gap.yaml",
        "config/nowcast/diagnostic.yaml",
    ]
    legacy = [
        "config/legacy/nowcast/ff5_lag_production.yaml",
        "config/legacy/nowcast/market_only_all_candidates_backtest.yaml",
        "config/legacy/experiments/research.yaml",
    ]

    for path in active + legacy:
        assert Path(path).exists()
        load_config(path)


def test_readme_command_paths_and_start_script_exist() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for path in [
        "config/nowcast/production.yaml",
        "config/nowcast/backtest_release_gap.yaml",
        "config/nowcast/diagnostic.yaml",
    ]:
        assert path in readme
        assert Path(path).exists()
    assert Path("start_nowcast.sh").exists()
