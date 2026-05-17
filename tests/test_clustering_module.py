"""Tests unitaires pour `ml.clustering`.

Objectif : vérifier la détection conjointe DBSCAN + IsolationForest,
la robustesse sur petits datasets et la présence des colonnes attendues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.clustering import _normalize_series, _prepare_features, cluster_products


def _sample_cluster_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(n)],
            "title": [f"Produit {i}" for i in range(n)],
            "price": np.concatenate([rng.uniform(50, 300, n - 1), np.array([5000.0])]),
            "review_count": np.concatenate([rng.integers(10, 500, n - 1), np.array([1])]),
            "rating": np.concatenate([rng.uniform(3.0, 5.0, n - 1), np.array([0.2])]),
            "log_price": np.log1p(np.concatenate([rng.uniform(50, 300, n - 1), np.array([5000.0])])),
            "rating_score": rng.uniform(0.4, 1.0, n),
            "popularity_score": rng.uniform(0.2, 1.0, n),
            "discount_score": rng.uniform(0.0, 0.8, n),
            "category": rng.choice(["Mode", "Électronique", "Maison"], n),
            "shop_name": rng.choice(["ShopA", "ShopB", "ShopC"], n),
            "price_to_median_ratio": rng.uniform(0.7, 1.3, n),
        }
    )


def test_cluster_products_adds_anomaly_columns():
    df = cluster_products(_sample_cluster_df())

    for col in [
        "cluster",
        "cluster_label",
        "dbscan_cluster",
        "dbscan_outlier",
        "iforest_outlier",
        "price_anomaly_score",
        "is_anomaly",
        "PC1",
        "PC2",
        "pca_variance_explained",
    ]:
        assert col in df.columns

    assert df["dbscan_outlier"].isin([0, 1]).all()
    assert df["iforest_outlier"].isin([0, 1]).all()
    assert df["is_anomaly"].isin([0, 1]).all()
    assert df["price_anomaly_score"].between(0, 1).all()


def test_cluster_products_small_dataset_is_safe():
    df = _sample_cluster_df(n=5)
    out = cluster_products(df)

    assert len(out) == len(df)
    assert out["dbscan_outlier"].eq(0).all()
    assert out["is_anomaly"].isin([0, 1]).all()
    assert out["price_anomaly_score"].between(0, 1).all()


def test_extreme_row_is_ranked_more_anomalous():
    df = _sample_cluster_df(n=20)
    out = cluster_products(df)
    extreme_idx = out["price"].idxmax()
    assert out.loc[extreme_idx, "price_anomaly_score"] >= out["price_anomaly_score"].median()


def test_normalize_series_uses_historical_min_max():
    series = pd.Series([10.0, 20.0, 30.0], name="log_price")
    historical_min_max = {"log_price": {"min": 0.0, "max": 100.0}}

    normalized = _normalize_series(series, historical_min_max=historical_min_max)

    assert pytest.approx(normalized.iloc[0], rel=1e-6) == 0.10
    assert pytest.approx(normalized.iloc[1], rel=1e-6) == 0.20
    assert pytest.approx(normalized.iloc[2], rel=1e-6) == 0.30


def test_prepare_features_uses_historical_min_max():
    summary = pd.DataFrame(
        {
            "log_price": [10.0, 30.0],
            "rating_score": [2.0, 4.0],
            "popularity_score": [5.0, 15.0],
            "discount_score": [1.0, 3.0],
        },
        index=[0, 1],
    )
    historical_min_max = {
        "log_price": {"min": 0.0, "max": 100.0},
        "rating_score": {"min": 0.0, "max": 5.0},
        "popularity_score": {"min": 0.0, "max": 20.0},
        "discount_score": {"min": 0.0, "max": 4.0},
    }

    profiles = _prepare_features(summary, historical_min_max=historical_min_max)

    assert pytest.approx(profiles.loc[0, "log_price"], rel=1e-6) == 0.10
    assert pytest.approx(profiles.loc[1, "log_price"], rel=1e-6) == 0.30
    assert pytest.approx(profiles.loc[0, "rating_score"], rel=1e-6) == 0.40
    assert pytest.approx(profiles.loc[1, "discount_score"], rel=1e-6) == 0.75


def test_cluster_products_saves_artifacts(tmp_path):
    df = _sample_cluster_df(n=30)
    scaler_path = tmp_path / "scaler.joblib"
    models_dir = tmp_path / "models"

    out = cluster_products(df, scaler_path=str(scaler_path), models_dir=str(models_dir))

    assert len(out) == len(df)
    assert scaler_path.exists()
    for name in ["kmeans.joblib", "dbscan.joblib", "isolation_forest.joblib", "pca.joblib", "clustering_meta.joblib"]:
        assert (models_dir / name).exists()

    import joblib

    meta = joblib.load(models_dir / "clustering_meta.joblib")
    assert "historical_min_max" in meta
    assert "log_price" in meta["historical_min_max"]


def test_cluster_products_inference_uses_loaded_artifacts_without_fit(tmp_path, monkeypatch):
    df_train = _sample_cluster_df(n=30)
    df_test = _sample_cluster_df(n=25)
    scaler_path = tmp_path / "scaler.joblib"
    models_dir = tmp_path / "models"

    cluster_products(df_train, scaler_path=str(scaler_path), models_dir=str(models_dir))

    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    def _boom(*args, **kwargs):
        raise AssertionError("fit should not be called in inference mode")

    monkeypatch.setattr(StandardScaler, "fit", _boom)
    monkeypatch.setattr(KMeans, "fit", _boom)
    monkeypatch.setattr(DBSCAN, "fit", _boom)
    monkeypatch.setattr(IsolationForest, "fit", _boom)
    monkeypatch.setattr(PCA, "fit", _boom)

    from ml import clustering as clustering_module

    calls: list[dict[str, dict[str, float]] | None] = []
    original_normalize = clustering_module._normalize_series

    def _spy_normalize(series, historical_min_max=None):
        calls.append(historical_min_max)
        return original_normalize(series, historical_min_max=historical_min_max)

    monkeypatch.setattr(clustering_module, "_normalize_series", _spy_normalize)

    out = cluster_products(df_test, scaler_path=str(scaler_path), models_dir=str(models_dir))

    assert len(out) == len(df_test)
    for col in ["dbscan_outlier", "iforest_outlier", "price_anomaly_score", "is_anomaly"]:
        assert col in out.columns
    assert out["price_anomaly_score"].between(0, 1).all()
    assert any(call is not None and "log_price" in call for call in calls)


