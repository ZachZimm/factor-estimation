from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedFeatureExtractor:
    method: str
    feature_columns_in: list[str]
    feature_columns_out: list[str]
    model: Any
    metadata: dict[str, Any]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.reindex(columns=self.feature_columns_in).astype(float)
        if self.method == "group_pca":
            return _transform_group_pca(self, frame)
        if self.method == "pls":
            values = self.model.transform(frame)
            return pd.DataFrame(values, index=frame.index, columns=self.feature_columns_out)
        if self.method == "clustered":
            return _transform_clustered(self, frame)
        raise ValueError(f"Unsupported feature extraction method: {self.method}")


@dataclass
class FittedPerTargetExtractor:
    method: str
    target_columns: list[str]
    feature_columns_in: list[str]
    feature_columns_by_target: dict[str, list[str]]
    models_by_target: dict[str, Any]
    metadata: dict[str, Any]

    def transform_for_target(self, X: pd.DataFrame, target: str) -> pd.DataFrame:
        if target not in self.models_by_target:
            raise ValueError(f"No fitted extractor for target: {target}")
        frame = X.reindex(columns=self.feature_columns_in).astype(float)
        values = self.models_by_target[target].transform(frame)
        return pd.DataFrame(values, index=frame.index, columns=self.feature_columns_by_target[target])


def should_apply_feature_extraction(config: dict, model_type: str) -> bool:
    cfg = config.get("feature_extraction", {})
    if not bool(cfg.get("enabled", False)):
        return False
    if cfg.get("method", "none") == "none":
        return False
    return model_type in set(cfg.get("apply_to_models", []))


def fit_feature_extractor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict,
    *,
    model_type: str,
) -> FittedFeatureExtractor | FittedPerTargetExtractor | None:
    if not should_apply_feature_extraction(config, model_type):
        return None
    method = str(config.get("feature_extraction", {}).get("method", "none"))
    if method == "group_pca":
        return _fit_group_pca(train_df, feature_columns, config)
    if method == "pls":
        return _fit_pls(train_df, feature_columns, target_columns, config)
    if method == "per_target_pls":
        return _fit_per_target_pls(train_df, feature_columns, target_columns, config)
    if method == "clustered":
        return _fit_clustered(train_df, feature_columns, config)
    raise ValueError(f"Unsupported feature_extraction.method='{method}'")


def transform_feature_frame(
    extractor: FittedFeatureExtractor | None,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if extractor is None:
        return X.copy(), list(X.columns)
    transformed = extractor.transform(X)
    if transformed.empty or len(transformed.columns) == 0:
        raise ValueError(f"Feature extraction produced no model features for method='{extractor.method}'")
    if bool(extractor.metadata.get("keep_original_features", False)):
        transformed = pd.concat([X[extractor.feature_columns_in].astype(float), transformed], axis=1)
    return transformed, list(transformed.columns)


def _fit_group_pca(train_df: pd.DataFrame, feature_columns: list[str], config: dict) -> FittedFeatureExtractor:
    cfg = config.get("feature_extraction", {}).get("group_pca", {})
    X = train_df[feature_columns].astype(float)
    assignments = _assign_groups(feature_columns, cfg.get("groups", {}))
    pipelines: dict[str, Pipeline] = {}
    output_columns_by_group: dict[str, list[str]] = {}
    input_columns_by_group: dict[str, list[str]] = {}
    explained: dict[str, list[float]] = {}
    for group, columns in assignments.items():
        if not columns:
            continue
        group_cfg = cfg.get("groups", {}).get(group, cfg.get("groups", {}).get("other", {}))
        n_components = min(int(group_cfg.get("n_components", 1)), len(columns), max(len(X) - 1, 0))
        if n_components <= 0:
            continue
        steps: list[tuple[str, Any]] = []
        if bool(cfg.get("scale_before_pca", True)):
            steps.append(("scaler", StandardScaler()))
        steps.append(("pca", PCA(n_components=n_components)))
        pipeline = Pipeline(steps)
        pipeline.fit(X[columns])
        out_cols = [f"fx_group_pca__{group}__pc{i:02d}" for i in range(1, n_components + 1)]
        pipelines[group] = pipeline
        output_columns_by_group[group] = out_cols
        input_columns_by_group[group] = columns
        explained[group] = [float(v) for v in pipeline.named_steps["pca"].explained_variance_ratio_]
    feature_columns_out = [col for cols in output_columns_by_group.values() for col in cols]
    if not feature_columns_out:
        raise ValueError("Feature extraction produced no model features for method='group_pca'")
    return FittedFeatureExtractor(
        method="group_pca",
        feature_columns_in=feature_columns,
        feature_columns_out=feature_columns_out,
        model={
            "pipelines": pipelines,
            "input_columns_by_group": input_columns_by_group,
            "output_columns_by_group": output_columns_by_group,
        },
        metadata={
            "method": "group_pca",
            "n_input_features": len(feature_columns),
            "n_output_features": len(feature_columns_out),
            "group_column_counts": {group: len(cols) for group, cols in input_columns_by_group.items()},
            "group_component_counts": {group: len(cols) for group, cols in output_columns_by_group.items()},
            "explained_variance_ratio": explained,
            "n_fit_rows": int(len(train_df)),
            "keep_original_features": bool(config.get("feature_extraction", {}).get("keep_original_features", False)),
        },
    )


def _transform_group_pca(extractor: FittedFeatureExtractor, X: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group, pipeline in extractor.model["pipelines"].items():
        columns = extractor.model["input_columns_by_group"][group]
        output_columns = extractor.model["output_columns_by_group"][group]
        values = pipeline.transform(X[columns])
        parts.append(pd.DataFrame(values, index=X.index, columns=output_columns))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)


def _fit_pls(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict,
) -> FittedFeatureExtractor:
    cfg = config.get("feature_extraction", {}).get("pls", {})
    X = train_df[feature_columns].astype(float)
    y = train_df[target_columns].astype(float)
    n_components = min(int(cfg.get("n_components", 20)), len(feature_columns), len(target_columns), max(len(X) - 1, 0))
    if n_components <= 0:
        raise ValueError("Feature extraction produced no model features for method='pls'")
    steps: list[tuple[str, Any]] = []
    if bool(cfg.get("scale_features", True)):
        steps.append(("scaler", StandardScaler()))
    steps.append(("pls", PLSRegression(n_components=n_components, scale=bool(cfg.get("scale_targets", False)))))
    pipeline = Pipeline(steps)
    pipeline.fit(X, y)
    feature_columns_out = [f"fx_pls__component_{i:02d}" for i in range(1, n_components + 1)]
    return FittedFeatureExtractor(
        method="pls",
        feature_columns_in=feature_columns,
        feature_columns_out=feature_columns_out,
        model=pipeline,
        metadata={
            "method": "pls",
            "n_input_features": len(feature_columns),
            "n_output_features": len(feature_columns_out),
            "n_components": n_components,
            "n_fit_rows": int(len(train_df)),
            "keep_original_features": bool(config.get("feature_extraction", {}).get("keep_original_features", False)),
        },
    )


def _fit_per_target_pls(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    config: dict,
) -> FittedPerTargetExtractor:
    cfg = config.get("feature_extraction", {}).get("per_target_pls", {})
    requested = int(cfg.get("n_components", 10))
    X = train_df[feature_columns].astype(float)
    models_by_target = {}
    feature_columns_by_target = {}
    component_counts = {}
    for target in target_columns:
        y = train_df[[target]].astype(float)
        n_components = min(requested, len(feature_columns), max(len(X) - 1, 0))
        last_error: Exception | None = None
        while n_components > 0:
            steps: list[tuple[str, Any]] = []
            if bool(cfg.get("scale_features", True)):
                steps.append(("scaler", StandardScaler()))
            steps.append(("pls", PLSRegression(n_components=n_components, scale=bool(cfg.get("scale_targets", False)))))
            pipeline = Pipeline(steps)
            try:
                pipeline.fit(X, y)
                break
            except ValueError as exc:
                last_error = exc
                n_components -= 1
        if n_components <= 0:
            raise ValueError(f"Unable to fit per-target PLS for {target}") from last_error
        columns = [f"fx_per_target_pls__{target}__component_{i:02d}" for i in range(1, n_components + 1)]
        models_by_target[target] = pipeline
        feature_columns_by_target[target] = columns
        component_counts[target] = n_components
    return FittedPerTargetExtractor(
        method="per_target_pls",
        target_columns=target_columns,
        feature_columns_in=feature_columns,
        feature_columns_by_target=feature_columns_by_target,
        models_by_target=models_by_target,
        metadata={
            "method": "per_target_pls",
            "n_input_features": len(feature_columns),
            "component_counts": component_counts,
            "n_fit_rows": int(len(train_df)),
        },
    )


def _fit_clustered(train_df: pd.DataFrame, feature_columns: list[str], config: dict) -> FittedFeatureExtractor:
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise ImportError("Install scipy>=1.11 to use feature_extraction.method='clustered'") from exc

    cfg = config.get("feature_extraction", {}).get("clustered", {})
    max_features = int(cfg.get("max_features_for_clustering", 2000))
    columns = feature_columns[:max_features]
    X = train_df[columns].astype(float)
    scaler = StandardScaler() if bool(cfg.get("scale_before_clustering", True)) else None
    values = scaler.fit_transform(X) if scaler else X.to_numpy()
    if len(columns) == 1:
        labels = np.asarray([1])
    else:
        distance = _safe_abs_correlation_distance(values)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        clusters = linkage(condensed, method="average")
        labels = fcluster(clusters, t=1.0 - float(cfg.get("correlation_threshold", 0.92)), criterion="distance")
    cluster_columns: dict[int, list[str]] = {}
    for label, column in zip(labels, columns):
        cluster_columns.setdefault(int(label), []).append(column)
    min_cluster = int(cfg.get("min_cluster_size", 2))
    keep_singletons = str(cfg.get("singleton_policy", "keep")) == "keep"
    retained = {
        label: cols
        for label, cols in sorted(cluster_columns.items())
        if len(cols) >= min_cluster or (len(cols) == 1 and keep_singletons)
    }
    if not retained:
        raise ValueError("Feature extraction produced no model features for method='clustered'")
    output_columns = [f"fx_clustered__cluster_{i:03d}" for i in range(1, len(retained) + 1)]
    return FittedFeatureExtractor(
        method="clustered",
        feature_columns_in=columns,
        feature_columns_out=output_columns,
        model={
            "scaler": scaler,
            "cluster_columns": list(retained.values()),
            "output_columns": output_columns,
        },
        metadata={
            "method": "clustered",
            "n_input_features": len(columns),
            "n_output_features": len(output_columns),
            "n_clusters": len(retained),
            "n_singletons": sum(1 for cols in retained.values() if len(cols) == 1),
            "correlation_threshold": float(cfg.get("correlation_threshold", 0.92)),
            "largest_clusters": sorted((cols for cols in retained.values()), key=len, reverse=True)[:10],
            "n_fit_rows": int(len(train_df)),
            "keep_original_features": bool(config.get("feature_extraction", {}).get("keep_original_features", False)),
        },
    )


def _transform_clustered(extractor: FittedFeatureExtractor, X: pd.DataFrame) -> pd.DataFrame:
    values = extractor.model["scaler"].transform(X) if extractor.model["scaler"] else X.to_numpy()
    standardized = pd.DataFrame(values, index=X.index, columns=extractor.feature_columns_in)
    data = {}
    for output_column, columns in zip(extractor.model["output_columns"], extractor.model["cluster_columns"]):
        data[output_column] = standardized[columns].mean(axis=1)
    return pd.DataFrame(data, index=X.index)


def _assign_groups(feature_columns: list[str], groups: dict[str, Any]) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {group: [] for group in groups}
    if "other" not in assignments:
        assignments["other"] = []
    ordered_groups = [group for group in groups if group != "other"] + ["other"]
    for feature in feature_columns:
        for group in ordered_groups:
            patterns = groups.get(group, {}).get("patterns", ["*"] if group == "other" else [])
            if any(fnmatch(feature, pattern) for pattern in patterns):
                assignments.setdefault(group, []).append(feature)
                break
    return assignments


def _safe_abs_correlation_distance(values: np.ndarray) -> np.ndarray:
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    numerator = centered.T @ centered
    denominator = np.outer(norms, norms)
    corr = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=corr, where=denominator > 0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return 1.0 - np.abs(corr)
