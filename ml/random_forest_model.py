"""Modèle RandomForest pour la classification Top-K."""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from loguru import logger

from ml.metrics import evaluate_classifier
from ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from configs.settings import TEST_SIZE, RANDOM_STATE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"


def train(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Entraîne et évalue un RandomForestClassifier.

    Stratégie class_weight='balanced' pour gérer le déséquilibre Top-K / reste.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("[RandomForest] Entraînement...")
    model.fit(X_train, y_train)
    evaluate_classifier(y_test, model.predict(X_test), model_name="RandomForest")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    logger.info(f"[RandomForest] Importance features : {importances}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.success(f"[RandomForest] → {MODEL_PATH}")
    return model


def load() -> RandomForestClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)