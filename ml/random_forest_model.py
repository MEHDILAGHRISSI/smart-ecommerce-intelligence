"""Modèle RandomForest pour la classification Top-K."""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from loguru import logger

from ml.metrics import evaluate_classifier
from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS, resolve_target_column, select_model_features
from ml.model_utils import optimal_f1_threshold, predict_with_threshold
from configs.settings import TEST_SIZE, RANDOM_STATE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"

LEAKAGE_COLUMNS = {
    "price",
    "rating",
    "review_count",
    "price_score",
    "rating_score",
    "log_reviews",
    "popularity_score",
    "value_for_money",
    "review_rating_coherence",
    "composite_score",
}


def _safe_stratify(y: pd.Series) -> pd.Series | None:
    return y if y.nunique() > 1 and int(y.value_counts().min()) >= 2 else None


def train(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Entraîne et évalue un RandomForestClassifier.

    Stratégie class_weight='balanced' pour gérer le déséquilibre Top-K / reste.
    """
    target_column = resolve_target_column(df, preferred_targets=TARGET_COLUMNS)
    feature_cols = select_model_features(
        df,
        candidate_columns=FEATURE_COLUMNS,
        target_column=target_column,
        leakage_columns=LEAKAGE_COLUMNS,
    )
    if not feature_cols:
        raise ValueError("Aucune feature exploitable disponible après exclusion des colonnes à risque.")

    X = df[feature_cols].fillna(0)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=_safe_stratify(y),
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=_safe_stratify(y_train),
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("[RandomForest] Entraînement...")
    model.fit(X_fit, y_fit)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    optimal_threshold, validation_f1 = optimal_f1_threshold(y_val, y_val_proba)

    model.fit(X_train, y_train)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred_optimal = predict_with_threshold(y_test_proba, optimal_threshold)

    metrics = evaluate_classifier(
        y_test,
        y_pred_optimal,
        y_proba=y_test_proba,
        model_name="RandomForest",
    )
    metrics["optimal_threshold"] = round(float(optimal_threshold), 4)
    metrics["validation_f1_at_threshold"] = round(float(validation_f1), 4)
    logger.info(
        f"[RandomForest] Seuil optimal={optimal_threshold:.4f} | F1 validation={validation_f1:.4f}"
    )

    importances = dict(zip(feature_cols, model.feature_importances_.round(4)))
    logger.info(f"[RandomForest] Importance features : {importances}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.success(f"[RandomForest] → {MODEL_PATH}")
    return model


def load() -> RandomForestClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)