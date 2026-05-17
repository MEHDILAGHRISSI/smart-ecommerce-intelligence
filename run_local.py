"""
run_local.py — Smart eCommerce Intelligence
============================================
FST Tanger — LSI2 — DM & SID 2025/2026

Pipeline ML local complet (sans Kubeflow) :
  Étape 1 → Nettoyage          (ml/cleaner.py)
  Étape 2 → Feature Engineering (20 features + composite_score)
  Étape 3 → Supervisé           (RandomForest + XGBoost)
  Étape 4 → Clustering          (KMeans + DBSCAN + PCA)
  Étape 5 → Règles d'association (Apriori)
  Étape 6 → Export Dashboard    (products_final.csv + top_k.csv)

Usage :
    python run_local.py
    python run_local.py --input data/raw/products_enriched_overnight.json
    python run_local.py --input data/raw/products_enriched_overnight.json --topk 50
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# ── sys.path : rend configs/ et ml/ importables ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from configs.settings import DATA_PROCESSED_DIR, DATA_RAW_DIR
from ml.cleaner import clean, load_latest_raw_products
from ml.clustering import cluster_products

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "price_score", "rating_score", "popularity_score", "stock_score", "discount_score",
    "value_for_money", "price_to_median_ratio", "log_reviews", "variant_diversity",
    "product_completeness", "has_discount_flag", "is_in_stock_flag", "n_images_norm",
    "brand_score", "shop_reputation", "review_rating_coherence",
    "title_length_norm", "category_popularity", "price_volatility", "tag_diversity",
]
CLUSTER_TARGET_LEAKAGE_COLUMNS = {
    "price_score",
    "rating_score",
    "popularity_score",
    "stock_score",
    "discount_score",
    "value_for_money",
    "price_to_median_ratio",
    "log_reviews",
    "variant_diversity",
    "n_images_norm",
    "log_price",
    "composite_score",
}
CLUSTER_FEATURES = ["log_price", "rating_score", "popularity_score", "stock_score", "discount_score"]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def _save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Sauvegarde un DataFrame dans data/processed/ et retourne le chemin."""
    out = DATA_PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.success(f"   💾 Sauvegardé → {out}  ({len(df)} lignes)")
    return out


def _step_banner(num: int, title: str) -> None:
    logger.info("")
    logger.info(f"{'═' * 60}")
    logger.info(f"  ÉTAPE {num} — {title}")
    logger.info(f"{'═' * 60}")


def _normalize_series(series: pd.Series) -> pd.Series:
    """Normalise une série en [0,1] de façon robuste."""
    s = pd.to_numeric(series, errors="coerce").astype(float).fillna(0.0)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    min_v = float(s.min())
    max_v = float(s.max())
    return (s - min_v) / (max_v - min_v)


def _assign_cluster_names(cluster_summary: pd.DataFrame) -> dict[int, str]:
    """Associe les clusters KMeans à des noms métier stables."""
    profiles = cluster_summary.copy()
    for col in ["log_price", "rating_score", "popularity_score", "stock_score", "discount_score"]:
        if col in profiles.columns:
            profiles[col] = _normalize_series(profiles[col])
        else:
            profiles[col] = 0.0

    score_map = {
        "Premium": profiles["log_price"] * 0.45 + profiles["rating_score"] * 0.35 + profiles["popularity_score"] * 0.20,
        "Budget": profiles["discount_score"] * 0.40 + (1 - profiles["log_price"]) * 0.35 + (1 - profiles["rating_score"]) * 0.25,
        "Populaire": profiles["popularity_score"] * 0.55 + profiles["rating_score"] * 0.25 + profiles["stock_score"] * 0.20,
        "Inactif": (1 - profiles["popularity_score"]) * 0.50 + (1 - profiles["stock_score"]) * 0.30 + (1 - profiles["rating_score"]) * 0.20,
    }

    remaining = list(profiles.index)
    mapping: dict[int, str] = {}
    for semantic_label in ["Premium", "Budget", "Populaire", "Inactif"]:
        score_series = score_map[semantic_label].astype(float)
        candidates = {int(idx): float(score_series.at[idx]) for idx in remaining}
        chosen_idx = max(candidates, key=candidates.get)
        mapping[chosen_idx] = semantic_label
        remaining.remove(chosen_idx)
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────
def step_clean(input_file: str | None) -> pd.DataFrame:
    _step_banner(1, "NETTOYAGE DES DONNÉES")

    if input_file:
        path = Path(input_file)
        logger.info(f"   📂 Fichier spécifié : {path.name}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        df_raw = pd.DataFrame(data)
        logger.info(f"   📥 {len(df_raw)} produits bruts chargés")
    else:
        logger.info(f"   📂 Chargement du JSON le plus récent dans {DATA_RAW_DIR}")
        df_raw = load_latest_raw_products()

    df_clean = clean(df_raw)
    _save_csv(df_clean, "products_cleaned.csv")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — FEATURE ENGINEERING (20 features)
# ─────────────────────────────────────────────────────────────────────────────
def step_features(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    _step_banner(2, "FEATURE ENGINEERING (20 features)")
    n = len(df)

    # ── Groupe 1 : Scores normalisés [0, 1] ───────────────────────────────────
    max_price = df["price"].max()
    df["price_score"] = (1 - df["price"] / max_price).clip(0, 1) if max_price > 0 else 0.5
    df["log_price"] = pd.Series(
        np.log1p(pd.to_numeric(df["price"], errors="coerce").clip(lower=0).fillna(0.0)),
        index=df.index,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["rating_score"] = (df["rating"] / 5.0).clip(0, 1)
    df["log_reviews"] = np.log1p(df["review_count"])
    max_log = df["log_reviews"].max()
    df["popularity_score"] = (df["log_reviews"] / max_log).clip(0, 1) if max_log > 0 else 0.0
    max_stock = df["stock"].max()
    df["stock_score"] = (df["stock"] / max_stock).clip(0, 1) if max_stock > 0 else 0.0

    disc_col = "discount_pct" if "discount_pct" in df.columns else "discount_percentage"
    if disc_col in df.columns:
        max_disc = df[disc_col].max()
        df["discount_score"] = (df[disc_col] / max_disc).clip(0, 1) if max_disc > 0 else 0.0
    else:
        df["discount_score"] = 0.0

    # ── Groupe 2 : Features dérivées ──────────────────────────────────────────
    df["value_for_money"] = (df["rating_score"] * df["price_score"]).clip(0, 1)
    df.loc[df["rating"] == 0, "value_for_money"] = 0.0

    cat_medians = df.groupby("category")["price"].transform("median")
    ratio = (df["price"] / cat_medians.replace(0, 1)).clip(0, 5)
    df["price_to_median_ratio"] = (1 - (ratio - 1).abs().clip(0, 4) / 4)

    if "n_variants" in df.columns:
        max_v = df["n_variants"].clip(upper=20).max()
        df["variant_diversity"] = (df["n_variants"].clip(upper=20) / max_v).clip(0, 1) if max_v > 0 else 0.0
    else:
        df["variant_diversity"] = 0.0

    if "n_images" in df.columns:
        max_img = df["n_images"].clip(upper=10).max()
        df["n_images_norm"] = (df["n_images"].clip(upper=10) / max_img).clip(0, 1) if max_img > 0 else 0.5
    else:
        df["n_images_norm"] = 0.5

    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts(normalize=True)
        df["brand_score"] = df["brand"].map(brand_counts).fillna(0.0).clip(0, 1)
    else:
        df["brand_score"] = 0.0

    # ── Groupe 3 : Features métier ────────────────────────────────────────────
    comp_cols = [c for c in ["price", "rating", "review_count", "stock", "category"] if c in df.columns]
    df["product_completeness"] = df[comp_cols].notna().mean(axis=1)
    df.loc[df["price"] == 0, "product_completeness"] *= 0.5

    disc_col2 = "has_discount" if "has_discount" in df.columns else None
    df["has_discount_flag"] = df[disc_col2].astype(float) if disc_col2 else (df["discount_score"] > 0).astype(float)
    df["is_in_stock_flag"] = df["is_in_stock"].astype(float) if "is_in_stock" in df.columns else (df["stock"] > 0).astype(float)

    if "shop_name" in df.columns:
        shop_avg_rating = df.groupby("shop_name")["rating"].transform("mean")
        df["shop_reputation"] = (shop_avg_rating / 5.0).clip(0, 1)
    else:
        df["shop_reputation"] = 0.5

    df["review_rating_coherence"] = (df["popularity_score"] * df["rating_score"]).clip(0, 1)

    # ── Groupe 4 : Features texte/diversité ───────────────────────────────────
    if "title" in df.columns:
        title_len = df["title"].str.len().fillna(0)
        df["title_length_norm"] = (title_len.clip(10, 100) - 10) / 90
    else:
        df["title_length_norm"] = 0.5

    cat_counts = df["category"].value_counts(normalize=True)
    df["category_popularity"] = df["category"].map(cat_counts).fillna(0.0).clip(0, 1)
    df["price_volatility"] = (ratio - 1.0).abs().clip(0, 4) / 4

    if "tags" in df.columns:
        n_tags = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        max_tags = n_tags.clip(upper=20).max()
        df["tag_diversity"] = (n_tags.clip(upper=20) / max_tags).clip(0, 1) if max_tags > 0 else 0.0
    else:
        df["tag_diversity"] = 0.0

    # ── Composite Score (ranking heuristique uniquement, pas une cible) ──────
    df["composite_score"] = (
        df["rating_score"] * 0.30
        + df["popularity_score"] * 0.25
        + df["discount_score"] * 0.20
        + df["value_for_money"] * 0.15
        + df["product_completeness"] * 0.10
    ).clip(0, 1)

    logger.info(f"   ✅ 20 features créées | Composite Score conservé comme score de ranking")
    _save_csv(df, "products_features.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — ENTRAÎNEMENT SUPERVISÉ (RandomForest + XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
def step_train(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    _step_banner(3, "ENTRAÎNEMENT SUPERVISÉ (RandomForest + XGBoost sur clusters)")

    if "cluster_label" not in df.columns:
        logger.warning("   ⚠️  Colonne cluster_label absente. Entraînement supervisé ignoré.")
        return {}

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns and c not in CLUSTER_TARGET_LEAKAGE_COLUMNS]
    X = df[feature_cols].fillna(0)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(df["cluster_label"].astype(str))
    if len(np.unique(y)) < 2:
        logger.warning("   ⚠️  Trop peu de clusters uniques pour entraîner un classifieur.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    metrics = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── RandomForest ──────────────────────────────────────────────────────────
    logger.info("   [RandomForest] Entraînement...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")

    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    precision_rf = precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    recall_rf = recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
    auc_rf = roc_auc_score(y_test, y_proba_rf, multi_class="ovr", average="weighted") if y_proba_rf.shape[1] > 2 else roc_auc_score(y_test, y_proba_rf[:, 1])
    metrics["rf"] = {
        "accuracy": round(acc_rf, 3),
        "f1": round(f1_rf, 3),
        "precision": round(precision_rf, 3),
        "recall": round(recall_rf, 3),
        "auc_roc": round(float(auc_rf), 3),
    }
    logger.info(f"   [RF] Acc={acc_rf:.3f} | F1={f1_rf:.3f} | AUC={float(auc_rf):.3f}")

    # Feature Importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    _save_csv(fi, "feature_importance.csv")
    logger.info(f"   [RF] Top-3 Features : {', '.join(fi.head(3)['feature'])}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    logger.info("   [XGBoost] Entraînement...")
    n_classes = len(le.classes_)
    xgb_kwargs = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 150,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
    }
    if n_classes > 2:
        xgb_kwargs["num_class"] = n_classes
    xgb = XGBClassifier(**xgb_kwargs)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0)
    precision_xgb = precision_score(y_test, y_pred_xgb, average="weighted", zero_division=0)
    recall_xgb = recall_score(y_test, y_pred_xgb, average="weighted", zero_division=0)
    auc_xgb = roc_auc_score(y_test, y_proba_xgb, multi_class="ovr", average="weighted") if y_proba_xgb.shape[1] > 2 else roc_auc_score(y_test, y_proba_xgb[:, 1])
    metrics["xgb"] = {
        "accuracy": round(acc_xgb, 3),
        "f1": round(f1_xgb, 3),
        "precision": round(precision_xgb, 3),
        "recall": round(recall_xgb, 3),
        "auc_roc": round(float(auc_xgb), 3),
    }
    logger.info(f"   [XGB] Acc={acc_xgb:.3f} | F1={f1_xgb:.3f} | AUC={float(auc_xgb):.3f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — CLUSTERING (KMeans + DBSCAN + PCA 2D)
# ─────────────────────────────────────────────────────────────────────────────
def step_cluster(df: pd.DataFrame) -> pd.DataFrame:
    _step_banner(4, "CLUSTERING (KMeans + DBSCAN + PCA 2D)")
    df = cluster_products(df, pca_components=2)
    _save_csv(df, "products_clustered.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — RÈGLES D'ASSOCIATION (Apriori)
# ─────────────────────────────────────────────────────────────────────────────
def step_apriori(df: pd.DataFrame) -> pd.DataFrame:
    _step_banner(5, "RÈGLES D'ASSOCIATION (Apriori)")
    rules_df = pd.DataFrame()

    if "shop_name" not in df.columns or "category" not in df.columns:
        logger.warning("   ⚠️  Colonnes shop_name/category manquantes. Apriori ignoré.")
        _save_csv(rules_df, "association_rules.csv")
        return rules_df

    transactions = (
        df.groupby("shop_name")["category"]
        .apply(lambda cats: list(cats.dropna().unique()))
        .tolist()
    )
    transactions = [t for t in transactions if len(t) >= 2]
    n_tx = len(transactions)
    logger.info(f"   Transactions (boutiques avec ≥2 catégories) : {n_tx}")

    if n_tx < 3:
        logger.warning("   ⚠️  Pas assez de transactions. Apriori ignoré.")
        _save_csv(rules_df, "association_rules.csv")
        return rules_df

    # Support adaptatif : au moins 2 boutiques doivent partager l'itemset
    min_sup = max(0.3, 2 / n_tx)
    logger.info(f"   min_support adaptatif = {min_sup:.2f} ({n_tx} transactions)")

    te = TransactionEncoder()
    basket_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

    # max_len=3 évite l'explosion combinatoire
    frequent = apriori(basket_df, min_support=min_sup, use_colnames=True, max_len=3)

    if not frequent.empty:
        rules_df = association_rules(frequent, metric="confidence", min_threshold=0.5)
        rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
        logger.info(f"   ✅ {len(rules_df)} règles extraites")
    else:
        logger.warning("   ⚠️  Aucun itemset fréquent trouvé avec ce support.")

    _save_csv(rules_df, "association_rules.csv")
    return rules_df

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — EXPORT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def step_export(df: pd.DataFrame, top_k: int) -> None:
    _step_banner(6, "EXPORT FINAL POUR LE DASHBOARD")

    # Sauvegarde du fichier principal
    _save_csv(df, "products_final.csv")

    # Génération du fichier PCA pour le dashboard
    # CORRECTION : Inclure toutes les colonnes nécessaires pour l'affichage des metrics
    pca_cols = [
        "PC1", "PC2",                      # Coordonnées PCA (obligatoires)
        "pca_variance_explained",          # Pour la metric card de variance
        "cluster", "cluster_label", "dbscan_outlier", "iforest_outlier",
        "is_anomaly", "price_anomaly_flag", "price_anomaly_score",
        "title", "category", "shop_name",  # Pour les hover data
        "is_top_product", "price", "rating", "composite_score"  # Données supplémentaires
    ]
    pca_cols_available = [c for c in pca_cols if c in df.columns]

    if "PC1" in df.columns and "PC2" in df.columns:
        _save_csv(df[pca_cols_available], "pca_viz.csv")
        logger.info("   🗺️  pca_viz.csv généré avec succès (colonnes PC1, PC2 + métadonnées)")
    else:
        logger.warning("   ⚠️  Colonnes PC1/PC2 manquantes. pca_viz.csv non généré.")

    # Génération du fichier Top-K
    top_products = df.nlargest(top_k, "composite_score")
    _save_csv(top_products, "top_k_products.csv")
    logger.info(f"   🏆 Top-{top_k} produits exportés")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(input_file: str | None, top_k: int) -> None:
    start = time.time()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Smart eCommerce Intelligence — Pipeline Local ML       ║")
    logger.info(f"║   FST Tanger — LSI2 — {datetime.now().strftime('%Y-%m-%d %H:%M')}                    ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    results = {}

    # ── Étape 1 : Nettoyage ──────────────────────────────────────────────────
    try:
        df = step_clean(input_file)
        results["step1_clean"] = f"✅ {len(df)} produits nettoyés"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 1 (Nettoyage) — ÉCHEC : {e}")
        raise SystemExit(1)

    # ── Étape 2 : Feature Engineering ────────────────────────────────────────
    try:
        df = step_features(df, top_k=top_k)
        results["step2_features"] = "✅ 20 features | ranking heuristique préparé"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 2 (Feature Engineering) — ÉCHEC : {e}")
        raise SystemExit(1)

    # ── Étape 3 : Clustering ──────────────────────────────────────────────────
    try:
        df = step_cluster(df)
        n_seg = df["cluster_label"].nunique() if "cluster_label" in df.columns else df["cluster"].nunique()
        results["step3_cluster"] = f"✅ {n_seg} segments métier | Isolation Forest + PCA 2D"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 3 (Clustering) — ÉCHEC : {e}")
        logger.warning("   Continuation avec les étapes suivantes...")

    # ── Étape 4 : Supervisé (apprend à prédire les clusters) ─────────────────
    try:
        ml_metrics = step_train(df)
        if ml_metrics:
            rf_auc = ml_metrics.get("rf", {}).get("auc_roc", "N/A")
            xgb_auc = ml_metrics.get("xgb", {}).get("auc_roc", "N/A")
            results["step4_train"] = f"✅ RF AUC={rf_auc} | XGB AUC={xgb_auc}"
        else:
            results["step4_train"] = "⚠️  Ignoré (clusters insuffisants)"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 4 (Supervisé clusters) — ÉCHEC : {e}")
        logger.warning("   Continuation avec les étapes suivantes...")

    # ── Étape 5 : Apriori ────────────────────────────────────────────────────
    try:
        rules_df = step_apriori(df)
        results["step5_apriori"] = f"✅ {len(rules_df)} règles d'association"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 5 (Apriori) — ÉCHEC : {e}")
        logger.warning("   Continuation avec les étapes suivantes...")

    # ── Étape 6 : Export ─────────────────────────────────────────────────────
    try:
        step_export(df, top_k=top_k)
        results["step6_export"] = "✅ CSV finaux prêts pour le dashboard"
    except Exception as e:
        logger.error(f"❌ ÉTAPE 6 (Export) — ÉCHEC : {e}")

    # ── Rapport final ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   RÉSUMÉ DU PIPELINE                                     ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    for step, status in results.items():
        logger.info(f"║  {step:<20} {status:<37}║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Durée totale : {elapsed:.1f}s                                    ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Fichiers générés dans data/processed/ :                 ║")
    for csv_file in sorted(DATA_PROCESSED_DIR.glob("*.csv")):
        size_kb = csv_file.stat().st_size // 1024
        logger.info(f"║    📄 {csv_file.name:<40} ({size_kb} KB)  ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Lancer le dashboard : streamlit run dashboard/app.py    ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")


def start_ml_pipeline(input_file: str | None = None, top_k: int = 20) -> None:
    """Point d'entrée interne pour l'orchestrateur et les appels programmatiques."""
    logger.info(f"🚀 Démarrage du pipeline ML interne sur : {input_file}")
    return main(input_file=input_file, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart eCommerce — Pipeline Local")
    parser.add_argument(
        "--input", default=None,
        help="Chemin vers le fichier JSON (défaut: fichier le plus récent dans data/raw/)"
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Nombre de produits Top-K à sélectionner (défaut: 20)"
    )
    args = parser.parse_args()
    start_ml_pipeline(args.input, args.topk)
