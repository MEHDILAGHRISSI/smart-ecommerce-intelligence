"""Clustering avancé et détection d'anomalies pour les produits e-commerce.

Ce module concentre la segmentation non supervisée et les flags d'outliers :
- KMeans pour la segmentation principale
- DBSCAN pour isoler le bruit dense
- IsolationForest pour les anomalies financières
- PCA 2D/3D pour la visualisation dashboard

La fonction principale `cluster_products` conserve une signature simple et
retourne le DataFrame enrichi sans casser le pipeline existant.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from configs.settings import KMEANS_N_CLUSTERS, RANDOM_STATE
from ml.metrics import evaluate_clustering

CLUSTER_FEATURES = ["log_price", "rating_score", "popularity_score", "discount_score"]
IFOREST_FEATURES = ["price", "review_count", "rating"]
SEMANTIC_LABELS = ["Premium", "Budget", "Populaire", "Inactif"]
MODEL_FILENAMES = {
    "scaler": "scaler.joblib",
    "kmeans": "kmeans.joblib",
    "dbscan": "dbscan.joblib",
    "iforest": "isolation_forest.joblib",
    "pca": "pca.joblib",
    "meta": "clustering_meta.joblib",
}



def _normalize_series(
    series: pd.Series,
    historical_min_max: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    """Normalise une série numérique dans [0, 1] de manière robuste.

    En mode inférence, les bornes historiques peuvent être fournies pour
    éviter de recalculer le min/max sur le lot courant.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float).fillna(0.0)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)

    min_v = float(s.min())
    max_v = float(s.max())

    series_name = series.name
    if historical_min_max and isinstance(series_name, str) and series_name in historical_min_max:
        bounds = historical_min_max[series_name] or {}
        try:
            hist_min = float(bounds["min"])
            hist_max = float(bounds["max"])
            if np.isfinite(hist_min) and np.isfinite(hist_max) and not np.isclose(hist_min, hist_max):
                min_v, max_v = hist_min, hist_max
        except (KeyError, TypeError, ValueError):
            pass

    if np.isclose(min_v, max_v):
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)


def _prepare_features(
    cluster_summary: pd.DataFrame,
    historical_min_max: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Prépare les features de profil de cluster pour le scoring métier."""
    profiles = cluster_summary.copy()
    for col in ["log_price", "rating_score", "popularity_score", "stock_score", "discount_score"]:
        if col in profiles.columns:
            profiles[col] = _normalize_series(profiles[col], historical_min_max=historical_min_max)
        else:
            profiles[col] = 0.0
    return profiles


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Retourne une série numérique ou une série par défaut alignée sur l'index."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def _matrix_from_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Construit une matrice numérique avec colonnes garanties et ordre stable."""
    matrix = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            matrix[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
        else:
            matrix[column] = 0.0
    return matrix


def _artifact_paths(scaler_path: str | None, models_dir: str | None) -> dict[str, Path] | None:
    """Résout les chemins d'artefacts si le mode persistant est demandé."""
    if not scaler_path and not models_dir:
        return None

    if models_dir:
        models_root = Path(models_dir)
    else:
        assert scaler_path is not None
        models_root = Path(scaler_path).resolve().parent

    scaler_file = Path(scaler_path) if scaler_path else models_root / MODEL_FILENAMES["scaler"]

    return {
        "root": models_root,
        "scaler": scaler_file,
        "kmeans": models_root / MODEL_FILENAMES["kmeans"],
        "dbscan": models_root / MODEL_FILENAMES["dbscan"],
        "iforest": models_root / MODEL_FILENAMES["iforest"],
        "pca": models_root / MODEL_FILENAMES["pca"],
        "meta": models_root / MODEL_FILENAMES["meta"],
    }


def _artifacts_exist(paths: dict[str, Path]) -> bool:
    required = ["scaler", "kmeans", "dbscan", "iforest", "pca", "meta"]
    return all(paths[key].exists() for key in required)


def _save_artifacts(
    paths: dict[str, Path],
    *,
    scaler: StandardScaler,
    kmeans: KMeans | None,
    dbscan: DBSCAN | None,
    iforest: IsolationForest | None,
    pca: PCA | None,
    meta: dict[str, Any],
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, paths["scaler"])
    if kmeans is not None:
        joblib.dump(kmeans, paths["kmeans"])
    if dbscan is not None:
        joblib.dump(dbscan, paths["dbscan"])
    if iforest is not None:
        joblib.dump(iforest, paths["iforest"])
    if pca is not None:
        joblib.dump(pca, paths["pca"])
    joblib.dump(meta, paths["meta"])
    logger.success(f"[Clustering] Artefacts sauvegardés dans {paths['root']}")


def _load_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "scaler": joblib.load(paths["scaler"]),
        "kmeans": joblib.load(paths["kmeans"]),
        "dbscan": joblib.load(paths["dbscan"]),
        "iforest": joblib.load(paths["iforest"]),
        "pca": joblib.load(paths["pca"]),
        "meta": joblib.load(paths["meta"]),
    }


def _assign_cluster_names(
    cluster_summary: pd.DataFrame,
    historical_min_max: dict[str, dict[str, float]] | None = None,
) -> dict[int, str]:
    """Associe les clusters à des labels métier interprétables."""
    profiles = _prepare_features(cluster_summary, historical_min_max=historical_min_max)

    score_map = {
        "Premium": profiles["log_price"] * 0.45 + profiles["rating_score"] * 0.35 + profiles["popularity_score"] * 0.20,
        "Budget": profiles["discount_score"] * 0.40 + (1 - profiles["log_price"]) * 0.35 + (1 - profiles["rating_score"]) * 0.25,
        "Populaire": profiles["popularity_score"] * 0.55 + profiles["rating_score"] * 0.25 + profiles["stock_score"] * 0.20,
        "Inactif": (1 - profiles["popularity_score"]) * 0.50 + (1 - profiles["stock_score"]) * 0.30 + (1 - profiles["rating_score"]) * 0.20,
    }

    remaining = list(profiles.index)
    mapping: dict[int, str] = {}
    for semantic_label in SEMANTIC_LABELS:
        if not remaining:
            break
        score_series = score_map[semantic_label].astype(float).reindex(remaining).fillna(-np.inf)
        chosen_idx = int(score_series.idxmax())
        mapping[chosen_idx] = semantic_label
        remaining.remove(chosen_idx)
    return mapping


def _prepare_output_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes de sortie si elles sont absentes."""
    defaults: dict[str, Any] = {
        "cluster": 0,
        "cluster_label": "Inconnu",
        "dbscan_cluster": 0,
        "dbscan_outlier": 0,
        "iforest_outlier": 0,
        "price_anomaly_score": 0.0,
        "price_anomaly_flag": 0,
        "is_anomaly": 0,
        "PC1": 0.0,
        "PC2": 0.0,
        "pca_variance_explained": 0.0,
        "composite_score": 0.0,
        "is_top_product": 0,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df


def _compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """Score de ranking métier conservant le comportement historique du pipeline."""
    base_rank = {
        "Premium": 0.92,
        "Populaire": 0.82,
        "Budget": 0.62,
        "Inactif": 0.25,
    }
    base = df.get("cluster_label", pd.Series("Inconnu", index=df.index)).map(base_rank).fillna(0.5)
    rating = _series_or_default(df, "rating_score")
    popularity = _series_or_default(df, "popularity_score")
    discount = _series_or_default(df, "discount_score")
    return (base + 0.08 * rating + 0.08 * popularity + 0.04 * discount).clip(0, 1)


def _fit_pca_from_scaled(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    n_components: int = 2,
) -> tuple[pd.DataFrame, float, PCA | None]:
    """Projette les données en PCA 2D/3D à partir d'une matrice déjà standardisée."""
    df = df.copy()
    if len(df) < 2 or X_scaled.shape[1] < 2:
        df["PC1"] = 0.0
        df["PC2"] = 0.0
        if n_components >= 3:
            df["PC3"] = 0.0
        df["pca_variance_explained"] = 0.0
        return df, 0.0, None
    n_fit_components = min(max(2, n_components), X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_fit_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1] if X_pca.shape[1] >= 2 else 0.0
    if n_components >= 3:
        df["PC3"] = X_pca[:, 2] if X_pca.shape[1] >= 3 else 0.0

    variance_explained = float(sum(pca.explained_variance_ratio_[:2]) * 100)
    df["pca_variance_explained"] = round(variance_explained, 2)
    logger.info(
        f"[PCA] Variance expliquée (PC1+PC2) : {variance_explained:.1f}% | "
        f"n_components={pca.n_components_}"
    )
    return df, variance_explained, pca


def _normalize_anomaly_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Normalise les scores d'anomalie en [0, 1] (1 = anomalie extrême)."""
    scores = np.asarray(raw_scores, dtype=float)
    if scores.size == 0:
        return scores
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if np.isclose(min_v, max_v):
        return np.zeros_like(scores, dtype=float)
    normalized = (scores - min_v) / (max_v - min_v)
    return np.asarray(normalized, dtype=float)


def _dbscan_predict(model: DBSCAN, X_scaled: np.ndarray) -> np.ndarray:
    """Approximation de predict() pour DBSCAN à partir des core samples."""
    if not hasattr(model, "components_") or model.components_.size == 0:
        return np.full(X_scaled.shape[0], -1, dtype=int)

    core_samples = np.asarray(model.components_, dtype=float)
    core_labels = np.asarray(model.labels_[model.core_sample_indices_], dtype=int)
    distances = pairwise_distances(X_scaled, core_samples)
    nearest_idx = distances.argmin(axis=1)
    nearest_dist = distances[np.arange(len(X_scaled)), nearest_idx]

    predicted = np.full(X_scaled.shape[0], -1, dtype=int)
    within_eps = nearest_dist <= float(model.eps)
    predicted[within_eps] = core_labels[nearest_idx[within_eps]]
    return predicted


def _build_cluster_summary(model: KMeans, feature_names: list[str]) -> pd.DataFrame:
    summary = pd.DataFrame(model.cluster_centers_, columns=feature_names)
    if "stock_score" not in summary.columns:
        summary["stock_score"] = 0.0
    return summary


def cluster_products(
    df: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    pca_components: int = 2,
    contamination: float = 0.05,
    random_state: int = RANDOM_STATE,
    scaler_path: str | None = None,
    models_dir: str | None = None,
    artifact_paths: str | Path | None = None,
) -> pd.DataFrame:
    """Enrichit un catalogue avec clustering + outliers + PCA.

    Args:
        df: DataFrame produit déjà nettoyé et enrichi en features.
        n_clusters: Nombre de clusters KMeans visé (par défaut: 4, borné à 3..4 si possible).
        dbscan_eps: Rayon DBSCAN.
        dbscan_min_samples: min_samples DBSCAN.
        pca_components: Nombre de composantes PCA à calculer.
        contamination: Taux attendu d'anomalies pour IsolationForest.
        random_state: Graine de reproductibilité.
        scaler_path: Chemin vers le scaler sauvegardé (mode inférence).
        models_dir: Répertoire des artefacts modèles.
        artifact_paths: Alias pour models_dir (compatibilité run_pipeline.py).

    Returns:
        DataFrame enrichi avec:
        - cluster, cluster_label
        - dbscan_cluster, dbscan_outlier
        - iforest_outlier
        - price_anomaly_score
        - is_anomaly
        - PC1/PC2[/PC3], pca_variance_explained
        - composite_score, is_top_product
    """
    # artifact_paths sert d'alias pour models_dir (rétrocompatibilité run_pipeline.py)
    if artifact_paths is not None and models_dir is None:
        models_dir = str(artifact_paths)
    df = df.copy()
    if df.empty:
        logger.warning("[Clustering] DataFrame vide : sortie par défaut.")
        return _prepare_output_defaults(df)

    df = _prepare_output_defaults(df)
    artifact_paths = _artifact_paths(scaler_path, models_dir)
    use_persistent_models = artifact_paths is not None and _artifacts_exist(artifact_paths)
    loaded_artifacts: dict[str, Any] | None = None
    historical_min_max: dict[str, dict[str, float]] | None = None
    if use_persistent_models and artifact_paths is not None:
        try:
            loaded_artifacts = _load_artifacts(artifact_paths)
            logger.info(f"[Clustering] Artefacts chargés depuis {artifact_paths['root']}")
        except Exception as exc:
            logger.warning(f"[Clustering] Chargement artefacts impossible, bascule en entraînement : {exc}")
            loaded_artifacts = None

    base_cluster_matrix = _matrix_from_columns(df, [c for c in CLUSTER_FEATURES if c in df.columns])
    cluster_features = list(base_cluster_matrix.columns)
    iforest_cols = [c for c in IFOREST_FEATURES if c in df.columns]
    scaler: StandardScaler | None = None
    kmeans: KMeans | None = None
    dbscan: DBSCAN | None = None
    iforest: IsolationForest | None = None
    pca_model: PCA | None = None

    if loaded_artifacts is not None:
        meta = loaded_artifacts["meta"] if isinstance(loaded_artifacts.get("meta"), dict) else {}
        cluster_features = list(meta.get("cluster_features", [c for c in CLUSTER_FEATURES if c in df.columns]))
        iforest_cols = list(meta.get("iforest_features", IFOREST_FEATURES))
        pca_features = list(meta.get("pca_features", cluster_features))
        historical_min_max = meta.get("historical_min_max") if isinstance(meta.get("historical_min_max"), dict) else None

        scaler = loaded_artifacts["scaler"]
        kmeans = loaded_artifacts["kmeans"]
        dbscan = loaded_artifacts["dbscan"]
        iforest = loaded_artifacts["iforest"]
        pca_model = loaded_artifacts["pca"]
        assert scaler is not None and kmeans is not None and dbscan is not None and iforest is not None and pca_model is not None

        # ── Inference uniquement : pas de fit ───────────────────────────────
        X_cluster = _matrix_from_columns(df, cluster_features)
        X_scaled = scaler.transform(X_cluster)
        df["cluster"] = kmeans.predict(X_scaled)

        cluster_summary = _build_cluster_summary(kmeans, cluster_features)
        semantic_map = _assign_cluster_names(cluster_summary, historical_min_max=historical_min_max)
        df["cluster_label"] = df["cluster"].map(semantic_map).fillna("Inconnu")
        df["composite_score"] = _compute_composite_score(df)

        dbscan_labels = _dbscan_predict(dbscan, X_scaled)
        df["dbscan_cluster"] = dbscan_labels
        df["dbscan_outlier"] = (dbscan_labels == -1).astype(int)
        evaluate_clustering(X_scaled, dbscan_labels, model_name="DBSCAN")

        X_iforest = _matrix_from_columns(df, iforest_cols)
        preds = iforest.predict(X_iforest)
        raw_scores = -iforest.decision_function(X_iforest)
        df["iforest_outlier"] = (preds == -1).astype(int)
        df["price_anomaly_score"] = pd.Series(_normalize_anomaly_scores(raw_scores), index=df.index).round(6)

        df["is_anomaly"] = ((df["dbscan_outlier"] == 1) | (df["iforest_outlier"] == 1)).astype(int)
        df["price_anomaly_flag"] = df["iforest_outlier"].astype(int)
        df["is_top_product"] = df["cluster_label"].isin(["Premium", "Populaire"]).astype(int)

        X_pca = pca_model.transform(X_scaled)
        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1] if X_pca.shape[1] >= 2 else 0.0
        if pca_components >= 3 and X_pca.shape[1] >= 3:
            df["PC3"] = X_pca[:, 2]
        df["pca_variance_explained"] = round(float(sum(getattr(pca_model, "explained_variance_ratio_", [0, 0])[:2]) * 100), 2)

        logger.info(
            f"[Clustering] Inference terminée : {len(df)} produits | anomalies={int(df['is_anomaly'].sum())}"
        )
        return df

    # ── KMeans : segmentation principale ─────────────────────────────────────
    if len(cluster_features) < 2:
        logger.warning("[KMeans] Pas assez de features valides. Segmentation désactivée.")
        df["cluster"] = 0
        df["cluster_label"] = "Inconnu"
        df["composite_score"] = _compute_composite_score(df)
        df["dbscan_cluster"] = 0
        df["dbscan_outlier"] = 0
        df["iforest_outlier"] = 0
        df["price_anomaly_score"] = 0.0
        df["is_anomaly"] = 0
        df["price_anomaly_flag"] = 0
        df, variance_explained, pca_model = _fit_pca_from_scaled(df, np.zeros((len(df), 2), dtype=float), n_components=pca_components)
        if artifact_paths is not None:
            logger.warning("[Clustering] Artefacts non sauvegardés : features de clustering insuffisantes.")
        logger.info(
            f"[Clustering] Pipeline terminé : {len(df)} produits | anomalies={int(df['is_anomaly'].sum())} | PCA={variance_explained:.1f}%"
        )
        return df
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(base_cluster_matrix)
        preferred_clusters = n_clusters or min(max(3, min(KMEANS_N_CLUSTERS, 4)), 4)
        k = min(max(1, int(preferred_clusters)), len(df))

        if k < 2:
            logger.warning("[KMeans] Moins de 2 lignes, cluster unique assigné.")
            df["cluster"] = 0
            df["cluster_label"] = "Inconnu"
            kmeans = None
        else:
            logger.info(f"[KMeans] Entraînement avec K={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
            labels = kmeans.fit_predict(X_scaled)
            df["cluster"] = labels

            assert kmeans is not None
            cluster_summary = _build_cluster_summary(kmeans, cluster_features)
            historical_min_max = {
                col: {"min": float(cluster_summary[col].min()), "max": float(cluster_summary[col].max())}
                for col in ["log_price", "rating_score", "popularity_score", "stock_score", "discount_score"]
                if col in cluster_summary.columns
            }
            semantic_map = _assign_cluster_names(cluster_summary, historical_min_max=historical_min_max)
            df["cluster_label"] = df["cluster"].map(semantic_map).fillna("Inconnu")

            evaluate_clustering(X_scaled, labels, model_name="KMeans")

        df["composite_score"] = _compute_composite_score(df)

    # ── DBSCAN : bruit dense / outliers structurels ───────────────────────────
    if len(df) < 10 or len(cluster_features) < 2:
        logger.warning("[DBSCAN] Dataset trop petit ou features insuffisantes : désactivé proprement.")
        df["dbscan_cluster"] = 0
        df["dbscan_outlier"] = 0
    else:
        X_dbscan = X_scaled
        logger.info(f"[DBSCAN] Détection du bruit (eps={dbscan_eps}, min_samples={dbscan_min_samples})...")
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1)
        assert dbscan is not None
        dbscan_labels = dbscan.fit_predict(X_dbscan)
        df["dbscan_cluster"] = dbscan_labels
        df["dbscan_outlier"] = (dbscan_labels == -1).astype(int)
        evaluate_clustering(X_dbscan, dbscan_labels, model_name="DBSCAN")
        logger.info(
            f"[DBSCAN] {int(df['dbscan_outlier'].sum())} outliers sur {len(df)} produits"
        )

    # ── Isolation Forest : anomalie de prix ciblée ────────────────────────────
    if len(df) < 2 or len(iforest_cols) < 2:
        logger.warning("[IsolationForest] Dataset trop petit ou variables insuffisantes : désactivé.")
        df["iforest_outlier"] = 0
        df["price_anomaly_score"] = 0.0
    else:
        X_iforest = _matrix_from_columns(df, iforest_cols)
        logger.info(
            f"[IsolationForest] Entraînement sur {iforest_cols} (contamination={contamination})..."
        )
        iforest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = iforest.fit_predict(X_iforest)
        raw_scores = -iforest.decision_function(X_iforest)
        normalized_scores = _normalize_anomaly_scores(raw_scores)
        df["iforest_outlier"] = (preds == -1).astype(int)
        df["price_anomaly_score"] = pd.Series(normalized_scores, index=df.index).round(6)
        logger.info(
            f"[IsolationForest] {int(df['iforest_outlier'].sum())} anomalies financières détectées"
        )

    # ── Consolidation ─────────────────────────────────────────────────────────
    df["is_anomaly"] = ((df["dbscan_outlier"] == 1) | (df["iforest_outlier"] == 1)).astype(int)
    df["price_anomaly_flag"] = df["iforest_outlier"].astype(int)

    # KMeans a déjà produit les labels ; on complète la logique métier historique
    if "composite_score" not in df.columns:
        df["composite_score"] = _compute_composite_score(df)

    df["is_top_product"] = df["cluster_label"].isin(["Premium", "Populaire"]).astype(int)
    df.loc[df["is_anomaly"] == 1, "composite_score"] = (
        df.loc[df["is_anomaly"] == 1, "composite_score"] * 0.9
    ).clip(0, 1)

    # ── PCA pour le dashboard ─────────────────────────────────────────────────
    df, variance_explained, pca_model = _fit_pca_from_scaled(df, X_scaled, n_components=pca_components)

    if artifact_paths is not None:
        assert scaler is not None
        meta = {
            "cluster_features": cluster_features,
            "iforest_features": iforest_cols,
            "pca_features": cluster_features,
            "historical_min_max": historical_min_max,
            "n_clusters": int(df["cluster"].nunique()) if "cluster" in df.columns else 0,
            "pca_components": pca_components,
            "dbscan_eps": dbscan_eps,
            "dbscan_min_samples": dbscan_min_samples,
            "contamination": contamination,
            "random_state": random_state,
            "pca_variance_explained": variance_explained,
        }
        _save_artifacts(
            artifact_paths,
            scaler=scaler,
            kmeans=kmeans,
            dbscan=dbscan,
            iforest=iforest,
            pca=pca_model,
            meta=meta,
        )

    logger.info(
        f"[Clustering] Pipeline terminé : {len(df)} produits | "
        f"anomalies={int(df['is_anomaly'].sum())} | PCA={variance_explained:.1f}%"
    )

    return df


__all__ = ["cluster_products"]

