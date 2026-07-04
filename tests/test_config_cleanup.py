from __future__ import annotations

from pathlib import Path

from ff5_predictor.config import load_config


def test_active_configs_load() -> None:
    active = [
        "config/nowcast/latest.yaml",
        "config/nowcast/backtest_release_gap.yaml",
        "config/nowcast/diagnostic.yaml",
    ]
    research = [
        "config/research/extraction_group_pca.yaml",
        "config/research/extraction_pls.yaml",
        "config/research/extraction_per_target_pls.yaml",
        "config/research/extraction_clustered.yaml",
        "config/research/extraction_tft_group_pca.yaml",
        "config/research/extraction_clustered_095.yaml",
        "config/research/extraction_clustered_098.yaml",
        "config/research/extraction_clustered_hybrid_098.yaml",
        "config/research/extraction_group_pca_hybrid.yaml",
        "config/research/elasticnet_market_only.yaml",
        "config/research/elasticnet_market_only_refined.yaml",
        "config/research/per_factor_elasticnet_market_only.yaml",
        "config/research/backtest_size_value.yaml",
        "config/research/latest_size_value.yaml",
    ]

    for path in active + research:
        assert Path(path).exists()
        load_config(path)

    assert not Path("config/legacy").exists()


def test_readme_command_paths_and_start_script_exist() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for path in [
        "config/nowcast/latest.yaml",
        "config/nowcast/backtest_release_gap.yaml",
        "config/nowcast/diagnostic.yaml",
    ]:
        assert path in readme
        assert Path(path).exists()
    assert "config/research/" in readme
    assert "scripts/research/" in readme
    assert not any(Path("config/nowcast").glob("extraction_*.yaml"))
    assert Path("start_nowcast.sh").exists()
    assert Path("scripts/research/start_feature_extraction.sh").exists()
    assert Path("scripts/research/start_clustered_hybrid_experiments.sh").exists()
    assert Path("scripts/research/start_elasticnet_experiment.sh").exists()
    assert Path("scripts/research/start_elasticnet_refined_experiment.sh").exists()
    assert Path("scripts/research/start_per_factor_elasticnet_experiment.sh").exists()
