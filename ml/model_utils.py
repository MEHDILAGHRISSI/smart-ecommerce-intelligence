"""Utilitaires de modélisation pour éviter les fuites de données et optimiser le seuil."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import precision_recall_curve


def optimal_f1_threshold(y_true: Sequence[int], y_proba: Sequence[float]) -> tuple[float, float]:
    """Retourne le seuil qui maximise le F1 via la courbe Precision-Recall.

    Parameters
    ----------
    y_true:
        Labels binaires réels.
    y_proba:
        Probabilités prédites pour la classe positive.

    Returns
    -------
    tuple[float, float]
        Le meilleur seuil et le F1 associé.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_proba_arr)
    if thresholds.size == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def predict_with_threshold(y_proba: Sequence[float], threshold: float) -> np.ndarray:
    """Convertit des probabilités binaires en prédictions au seuil fourni."""
    return (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)

