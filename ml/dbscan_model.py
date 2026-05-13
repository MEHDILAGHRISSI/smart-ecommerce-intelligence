"""Modèle DBSCAN pour la détection d'anomalies (outliers) produits."""

from __future__ import annotations
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ml.metrics import evaluate_clustering

# DBSCAN travaille sur les données brutes (prix, note, avis) et non les scores normalisés
# pour détecter des anomalies réelles : ex. prix 10x la médiane, ou 0 avis avec note max
ANOMALY_FEATURES = ["price", "rating", "review_count"]


def detect_outliers(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    """
    Applique DBSCAN pour détecter les produits au profil atypique.

    Les produits avec label=-1 sont des outliers :
    - Prix aberrant (trop haut/bas par rapport aux voisins)
    - Note incohérente avec le nombre d'avis
    - Combinaison atypique qui mérite une vérification manuelle

    Args:
        eps: Rayon de voisinage. Ajuster selon la densité des données.
        min_samples: Nombre minimum de voisins pour former un cluster dense.

    Returns:
        DataFrame avec colonne 'dbscan_cluster' (-1 = outlier).
    """
    df = df.copy()
    features = [f for f in ANOMALY_FEATURES if f in df.columns]

    if len(features) < 2:
        logger.warning("[DBSCAN] Pas assez de features. Pas de détection d'anomalies.")
        df["dbscan_cluster"] = 0
        return df

    X_scaled = StandardScaler().fit_transform(df[features].values)

    logger.info(f"[DBSCAN] Détection anomalies (eps={eps}, min_samples={min_samples})...")
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)
    df["dbscan_cluster"] = labels

    n_outliers = (labels == -1).sum()
    n_clusters = len(set(labels) - {-1})
    logger.info(f"[DBSCAN] {n_clusters} clusters normaux | {n_outliers} outliers sur {len(df)} produits")

    evaluate_clustering(X_scaled, labels, model_name="DBSCAN")
    return df