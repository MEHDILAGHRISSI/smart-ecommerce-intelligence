"""Modèle XGBoost pour la classification Top-K."""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from loguru import logger

from ml.metrics import evaluate_classifier
from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMNS, resolve_target_column, select_model_features
from ml.model_utils import optimal_f1_threshold, predict_with_threshold
from configs.settings import TEST_SIZE, RANDOM_STATE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"

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


def train(df: pd.DataFrame) -> XGBClassifier:
    """Entraîne et évalue un XGBClassifier. Scale_pos_weight gère le déséquilibre."""
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

    neg, pos = (y_fit == 0).sum(), (y_fit == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )

    logger.info("[XGBoost] Entraînement...")
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    optimal_threshold, validation_f1 = optimal_f1_threshold(y_val, y_val_proba)

    model.fit(X_train, y_train, verbose=False)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_pred_optimal = predict_with_threshold(y_test_proba, optimal_threshold)

    metrics = evaluate_classifier(
        y_test,
        y_pred_optimal,
        y_proba=y_test_proba,
        model_name="XGBoost",
    )
    metrics["optimal_threshold"] = round(float(optimal_threshold), 4)
    metrics["validation_f1_at_threshold"] = round(float(validation_f1), 4)
    logger.info(f"[XGBoost] Seuil optimal={optimal_threshold:.4f} | F1 validation={validation_f1:.4f}")

    importances = dict(zip(feature_cols, model.feature_importances_.round(4)))
    logger.info(f"[XGBoost] Importance features : {importances}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.success(f"[XGBoost] → {MODEL_PATH}")
    return model


def load() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)