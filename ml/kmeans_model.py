"""Modèle non supervisé KMeans pour la segmentation des produits."""

from __future__ import annotations
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ml.metrics import evaluate_clustering
from configs.settings import KMEANS_N_CLUSTERS, RANDOM_STATE, MODELS_DIR

MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"
SCALER_PATH = MODELS_DIR / "kmeans_scaler.joblib"


def cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Applique l'algorithme KMeans pour segmenter les produits."""
    df = df.copy()

    # Sélection des variables pertinentes pour le clustering
    # On utilise les scores créés lors du feature engineering
    cluster_features = ["price_score", "rating_score", "popularity_score", "discount_score"]

    # Vérification que les colonnes existent
    features_to_use = [f for f in cluster_features if f in df.columns]

    if not features_to_use:
        logger.warning("[KMeans] Aucune feature numérique valide trouvée. Clustering annulé.")
        df["cluster"] = 0
        return df

    X = df[features_to_use]

    # Standardisation obligatoire pour le clustering basé sur la distance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement du modèle
    logger.info(f"[KMeans] Entraînement avec K={KMEANS_N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # Ajout des labels au DataFrame
    df["cluster"] = labels

    # Évaluation (Silhouette Score)
    evaluate_clustering(X_scaled, labels, model_name="KMeans")

    # Sauvegarde du modèle et du scaler
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.success(f"[KMeans] Modèle et Scaler sauvegardés dans {MODELS_DIR}")

    return df