"""Modèle XGBoost pour la classification Top-K."""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from loguru import logger

from ml.metrics import evaluate_classifier
from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from configs.settings import TEST_SIZE, RANDOM_STATE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"


def train(df: pd.DataFrame) -> XGBClassifier:
    """Entraîne et évalue un XGBClassifier. Scale_pos_weight gère le déséquilibre."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    evaluate_classifier(y_test, model.predict(X_test), model_name="XGBoost")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    logger.info(f"[XGBoost] Importance features : {importances}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.success(f"[XGBoost] → {MODEL_PATH}")
    return model


def load() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)