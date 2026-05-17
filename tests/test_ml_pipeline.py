"""Tests unitaires — Pipeline ML (structure aplatie ml/)."""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pytest

# Imports centralisés depuis ton module ML
from ml.cleaner import clean
from ml.feature_engineering import add_scoring_features, FEATURE_COLUMNS, select_model_features
from ml.model_utils import optimal_f1_threshold, predict_with_threshold
from ml.metrics import evaluate_classifier, evaluate_clustering


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(n)],
        "title": [f"Produit {i}" for i in range(n)],
        "price": np.random.uniform(50, 2000, n),
        "original_price": np.random.uniform(100, 2500, n),
        "rating": np.random.uniform(0, 5, n),
        "review_count": np.random.randint(0, 500, n),
        "stock": np.random.randint(0, 100, n),
        "category": np.random.choice(["Électronique", "Mode", "Maison", "Sport"], n),
        "source_platform": np.random.choice(["shopify", "woocommerce"], n),
        "shop_name": np.random.choice(["ShopA", "ShopB", "ShopC"], n),
    })


@pytest.fixture
def df_nulls(sample_df):
    df = sample_df.copy()
    df.loc[0:4, "price"] = 0
    df.loc[5:9, "rating"] = np.nan
    df.loc[10:14, "stock"] = np.nan
    df.loc[15, "title"] = ""
    return df


@pytest.fixture
def raw_json(tmp_path, sample_df):
    p = tmp_path / "products_test.json"
    p.write_text(json.dumps(sample_df.to_dict("records"), default=str), encoding="utf-8")
    return p


# ==========================================
# Tests : ml.cleaner
# ==========================================

def test_clean_doublons(sample_df):
    dupe = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    assert len(clean(dupe)) < len(dupe)


def test_clean_titres(df_nulls):
    c = clean(df_nulls)
    assert c["title"].notna().all()
    assert (c["title"].str.strip() != "").all()


def test_clean_prix(df_nulls):
    assert (clean(df_nulls)["price"] > 0).all()


def test_clean_discount(sample_df):
    c = clean(sample_df)
    assert "discount_percentage" in c.columns
    assert (c["discount_percentage"] >= 0).all()


# ==========================================
# Tests : ml.feature_engineering
# ==========================================

def test_features_0_1(sample_df):
    df = add_scoring_features(clean(sample_df))
    bounded_cols = [c for c in FEATURE_COLUMNS if c != "log_reviews"]
    for col in bounded_cols:
        assert df[col].between(0, 1).all(), f"{col} hors de l'intervalle [0,1]"
    assert (df["log_reviews"] >= 0).all()


def test_no_target_leakage(sample_df):
    df = add_scoring_features(clean(sample_df))
    assert "is_top_product" not in df.columns
    assert "composite_score" in df.columns


def test_select_model_features_excludes_sensitive_columns(sample_df):
    df = add_scoring_features(clean(sample_df))
    selected = select_model_features(
        df,
        target_column="is_top_product",
        leakage_columns={"rating", "review_count", "popularity_score", "composite_score"},
    )
    assert "is_top_product" not in selected
    assert "popularity_score" not in selected
    assert "rating_score" in selected


def test_optimal_f1_threshold_prefers_best_cutoff():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.35, 0.55, 0.95])
    threshold, best_f1 = optimal_f1_threshold(y_true, y_proba)
    y_pred_optimal = predict_with_threshold(y_proba, threshold)

    assert 0.0 <= threshold <= 1.0
    assert best_f1 >= 0.5
    assert y_pred_optimal.shape == y_true.shape


# ==========================================
# Tests : ml.metrics
# ==========================================

def test_classifier_keys():
    r = evaluate_classifier([0, 1, 0, 1], [0, 1, 0, 1])
    for k in ["accuracy", "f1_weighted", "precision", "recall"]:
        assert k in r


def test_clustering_silhouette():
    X = np.asarray(np.random.rand(30, 4), dtype=float)
    labels = np.array([0]*10 + [1]*10 + [2]*10)
    r = evaluate_clustering(X, labels)
    assert -1 <= r["silhouette_score"] <= 1


def test_clustering_1_cluster():
    r = evaluate_clustering(np.asarray(np.random.rand(20, 4), dtype=float), np.zeros(20, dtype=int))
    assert r["silhouette_score"] == -1.0


# ==========================================
# Smoke Test : Pipeline partiel
# ==========================================

def test_pipeline_smoke(sample_df):
    df = add_scoring_features(clean(sample_df))
    assert "composite_score" in df.columns
    assert "is_top_product" not in df.columns
