"""
Analyse PCA (Principal Component Analysis) — Réduction dimensionnelle 2D.

Objectifs :
- Visualiser les clusters dans un espace 2D compréhensible
- Calculer la variance expliquée par les 2 premières composantes
- Sauvegarder la projection pour le dashboard Streamlit
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from loguru import logger

from ml.feature_engineering import FEATURE_COLUMNS
from configs.settings import DATA_PROCESSED_DIR


def compute_pca(df: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, PCA]:
    """
    Applique PCA sur les features ML et retourne la projection 2D.

    Args:
        df: DataFrame avec les features FEATURE_COLUMNS.
        n_components: Nombre de composantes principales (2 pour visualisation).

    Returns:
        Tuple (DataFrame avec colonnes PC1, PC2, variance_explained), objet PCA.
    """
    features = [f for f in FEATURE_COLUMNS if f in df.columns]
    if len(features) < 2:
        logger.warning("[PCA] Pas assez de features pour PCA")
        df["PC1"] = 0.0
        df["PC2"] = 0.0
        df["pca_variance_explained"] = 0.0
        return df, None

    X = df[features].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    df = df.copy()
    for i in range(X_pca.shape[1]):
        df[f"PC{i+1}"] = X_pca[:, i]

    variance_explained = sum(pca.explained_variance_ratio_[:2]) * 100
    df["pca_variance_explained"] = round(variance_explained, 2)

    logger.success(
        f"[PCA] Variance expliquée (PC1+PC2) : {variance_explained:.1f}% | "
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}% | "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
    )

    # Loadings : quelles features contribuent le plus à chaque composante
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    ).round(4)
    logger.info(f"[PCA] Loadings :\n{loadings.sort_values('PC1', ascending=False)}")

    # Sauvegarde
    pca_path = DATA_PROCESSED_DIR / "pca_viz.csv"
    pca_path.parent.mkdir(parents=True, exist_ok=True)
    cols_to_save = [c for c in ["title", "category", "shop_name", "source_platform",
                                  "price", "rating", "composite_score", "cluster",
                                  "is_top_product", "PC1", "PC2"] if c in df.columns]
    df[cols_to_save].to_csv(pca_path, index=False)
    logger.success(f"[PCA] Projection 2D sauvegardée → {pca_path}")

    return df, pca