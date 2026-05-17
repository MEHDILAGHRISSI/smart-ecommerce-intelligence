"""Modèle DBSCAN pour la détection d'anomalies (outliers) produits."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from loguru import logger

from ml.metrics import evaluate_clustering

# Features utilisées pour la détection d'anomalies
ANOMALY_FEATURES = ["price", "rating", "review_count"]


def _knn_optimal_eps(
    X_scaled: np.ndarray,
    k: int = 5,
    window: int = 5
) -> float:
    """
    Calcule un eps optimal via la méthode du coude (elbow) sur les k-distances,
    avec lissage par moyenne mobile pour réduire le bruit sur données réelles.

    Args:
        X_scaled: Données normalisées
        k: Nombre de plus proches voisins (min_samples recommandé)
        window: Taille de la fenêtre de lissage (moyenne mobile)

    Returns:
        Valeur optimale de eps
    """
    # Calcul des k plus proches voisins
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)

    # On prend la distance au k-ième voisin
    k_distances = np.sort(distances[:, -1])

    # === LISSAGE par moyenne mobile ===
    if len(k_distances) >= window:
        # Moyenne mobile avec np.convolve
        kernel = np.ones(window) / window
        smoothed = np.convolve(k_distances, kernel, mode='valid')

        # Ajustement de l'index pour compenser le décalage de la convolution
        offset = window // 2
    else:
        smoothed = k_distances
        offset = 0

    # Calcul des dérivées sur la courbe lissée
    diffs_1 = np.diff(smoothed)
    diffs_2 = np.diff(diffs_1)

    # Détection du point de coude (changement le plus fort de concavité)
    if len(diffs_2) > 0:
        elbow_idx = int(np.argmax(np.abs(diffs_2))) + 1 + offset
        elbow_idx = min(elbow_idx, len(k_distances) - 1)
        optimal_eps = float(k_distances[elbow_idx])
    else:
        # Fallback : médiane des k-distances
        optimal_eps = float(np.median(k_distances))

    logger.info(f"[DBSCAN] eps optimal calculé = {optimal_eps:.4f} "
                f"(k={k}, window={window}, {len(k_distances)} points)")

    return optimal_eps


def detect_outliers(
    df: pd.DataFrame,
    min_samples: int = 5,
    use_auto_eps: bool = True
) -> pd.DataFrame:
    """
    Applique DBSCAN avec eps calculé dynamiquement via KNN lissé.

    Args:
        df: DataFrame contenant les produits
        min_samples: Nombre minimum de points pour former un cluster
        use_auto_eps: Si True, calcule eps automatiquement (recommandé)

    Returns:
        DataFrame avec colonne 'dbscan_cluster' (-1 = outlier)
    """
    df = df.copy()
    features = [f for f in ANOMALY_FEATURES if f in df.columns]

    if len(features) < 2 or len(df) < 10:
        logger.warning("[DBSCAN] Pas assez de données ou features. Anomalies désactivées.")
        df["dbscan_cluster"] = 0
        return df

    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    if use_auto_eps:
        eps = _knn_optimal_eps(X_scaled, k=min_samples, window=5)
    else:
        eps = 0.5  # valeur par défaut legacy

    logger.info(f"[DBSCAN] Détection anomalies (eps={eps:.4f}, min_samples={min_samples})...")

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    df["dbscan_cluster"] = labels

    n_outliers = (labels == -1).sum()
    n_clusters = len(set(labels) - {-1})

    logger.info(f"[DBSCAN] {n_clusters} clusters normaux | {n_outliers} outliers sur {len(df)} produits")

    evaluate_clustering(X_scaled, labels, model_name="DBSCAN")

    return df