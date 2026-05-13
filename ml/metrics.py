"""Métriques complètes : Accuracy, F1, AUC-ROC, CV, Silhouette, Davies-Bouldin."""
from __future__ import annotations
import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    silhouette_score, davies_bouldin_score,
)
from sklearn.model_selection import cross_val_score


def evaluate_classifier(y_true, y_pred, y_proba=None, model=None, X=None,
                        model_name: str = "Modèle", cv_folds: int = 5) -> dict:
    """Accuracy, F1, Précision, Rappel, AUC-ROC, CV-F1."""
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "auc_roc": None, "cv_f1_mean": None, "cv_f1_std": None,
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            pass
    if model is not None and X is not None:
        try:
            cv = cross_val_score(model, X, y_true, cv=cv_folds, scoring="f1_weighted", n_jobs=-1)
            metrics["cv_f1_mean"] = round(float(cv.mean()), 4)
            metrics["cv_f1_std"] = round(float(cv.std()), 4)
        except Exception:
            pass

    logger.info(f"\n{'=' * 55}\nÉvaluation : {model_name}")
    logger.info(f"  Accuracy={metrics['accuracy']} | F1={metrics['f1_weighted']} | "
                f"AUC-ROC={metrics['auc_roc']} | CV-F1={metrics['cv_f1_mean']}")
    logger.info(f"\n{metrics['classification_report']}\n{'=' * 55}")
    return metrics


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray,
                        model_name: str = "Clustering",
                        pca_variance: float | None = None) -> dict:
    """Silhouette Score + Davies-Bouldin Index."""
    unique = np.unique(labels)
    n_clusters = len(unique[unique != -1])
    metrics = {
        "model": model_name, "n_clusters": n_clusters,
        "n_outliers": int((labels == -1).sum()),
        "silhouette_score": None, "davies_bouldin": None,
        "pca_variance_explained": pca_variance,
    }
    if n_clusters >= 2:
        mask = labels != -1
        Xc = X_scaled[mask];
        lc = labels[mask]
        if len(Xc) > 1:
            try:
                metrics["silhouette_score"] = round(float(silhouette_score(Xc, lc)), 4)
            except Exception:
                metrics["silhouette_score"] = -1.0
            try:
                metrics["davies_bouldin"] = round(float(davies_bouldin_score(Xc, lc)), 4)
            except Exception:
                pass
    else:
        metrics["silhouette_score"] = -1.0
        logger.warning(f"[{model_name}] {n_clusters} cluster(s) — métriques invalides")

    logger.info(f"[{model_name}] K={n_clusters} | Silhouette={metrics['silhouette_score']} | "
                f"Davies-Bouldin={metrics['davies_bouldin']} | Outliers={metrics['n_outliers']}")
    return metrics