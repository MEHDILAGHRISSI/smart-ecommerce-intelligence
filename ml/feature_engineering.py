"""
Feature Engineering étendu — 20 features pour le pipeline ML.
Remplace ml/feature_engineering.py (13 features → 20 features).

Groupes :
  Groupe 1 — Scores normalisés [0,1]     : 5 features
  Groupe 2 — Features dérivées           : 6 features (+ brand_score)
  Groupe 3 — Features métier             : 5 features (+ shop_reputation, review_rating_coherence)
  Groupe 4 — Features texte/diversité   : 4 features (title_length, category_popularity,
                                                        price_volatility, tag_diversity)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger
from configs.settings import TOP_K, DATA_PROCESSED_DIR

FEATURE_COLUMNS = [
    # Groupe 1 : Scores normalisés
    "price_score",
    "rating_score",
    "popularity_score",
    "stock_score",
    "discount_score",
    # Groupe 2 : Features dérivées
    "value_for_money",
    "price_to_median_ratio",
    "log_reviews",
    "variant_diversity",
    "n_images_norm",
    "brand_score",           # NOUVEAU
    # Groupe 3 : Features métier
    "product_completeness",
    "has_discount_flag",
    "is_in_stock_flag",
    "shop_reputation",       # NOUVEAU
    "review_rating_coherence",  # NOUVEAU
    # Groupe 4 : Features texte/diversité
    "title_length_norm",     # NOUVEAU
    "category_popularity",   # NOUVEAU
    "price_volatility",      # NOUVEAU
    "tag_diversity",         # NOUVEAU
]

TARGET_COLUMN = "is_top_product"


def add_scoring_features(df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Construit 20 features ML + label is_top_product."""
    df = df.copy()
    n = len(df)

    # ── Groupe 1 : Scores normalisés [0,1] ───────────────────────────────────
    max_price = df["price"].max()
    df["price_score"] = (1 - df["price"] / max_price).clip(0, 1) if max_price > 0 else 0.5

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

    # ── Groupe 2 : Features dérivées ─────────────────────────────────────────
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

    # NOUVEAU — brand_score : popularité relative de la marque dans le catalogue
    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts(normalize=True)
        df["brand_score"] = df["brand"].map(brand_counts).fillna(0.0).clip(0, 1)
    else:
        df["brand_score"] = 0.0

    # ── Groupe 3 : Features métier ────────────────────────────────────────────
    comp_cols = [c for c in ["price", "rating", "review_count", "stock", "category", "image_url"] if c in df.columns]
    df["product_completeness"] = df[comp_cols].notna().mean(axis=1)
    df.loc[df["price"] == 0, "product_completeness"] *= 0.5

    disc_col2 = "has_discount" if "has_discount" in df.columns else None
    df["has_discount_flag"] = df[disc_col2].astype(float) if disc_col2 else (df["discount_score"] > 0).astype(float)

    df["is_in_stock_flag"] = df["is_in_stock"].astype(float) if "is_in_stock" in df.columns else (df["stock"] > 0).astype(float)

    # NOUVEAU — shop_reputation : note moyenne des produits du même shop
    if "shop_name" in df.columns:
        shop_avg_rating = df.groupby("shop_name")["rating"].transform("mean")
        df["shop_reputation"] = (shop_avg_rating / 5.0).clip(0, 1)
    else:
        df["shop_reputation"] = 0.5

    # NOUVEAU — review_rating_coherence : produits populaires ET bien notés
    df["review_rating_coherence"] = (df["popularity_score"] * df["rating_score"]).clip(0, 1)

    # ── Groupe 4 : Features texte / diversité ─────────────────────────────────
    # NOUVEAU — title_length_norm : titre ni trop court ni trop long → proxy de qualité
    if "title" in df.columns:
        title_len = df["title"].str.len().fillna(0)
        df["title_length_norm"] = ((title_len.clip(10, 100) - 10) / 90).clip(0, 1)
    else:
        df["title_length_norm"] = 0.5

    # NOUVEAU — category_popularity : fréquence de la catégorie = proxy de marché actif
    cat_counts = df["category"].value_counts(normalize=True)
    df["category_popularity"] = df["category"].map(cat_counts).fillna(0.0).clip(0, 1)

    # NOUVEAU — price_volatility : écart relatif au prix médian de la catégorie
    df["price_volatility"] = (ratio - 1.0).abs().clip(0, 4) / 4

    # NOUVEAU — tag_diversity : richesse des mots-clés du produit
    if "tags" in df.columns:
        tag_counts_series = df["tags"].fillna("").apply(
            lambda x: len(str(x).split(",")) if x and str(x) != "nan" else 0
        )
        max_tags = tag_counts_series.clip(upper=10).max()
        df["tag_diversity"] = (tag_counts_series.clip(upper=10) / max_tags).clip(0, 1) if max_tags > 0 else 0.0
    else:
        df["tag_diversity"] = 0.0

    # ── Score composite pondéré (20 features) ─────────────────────────────────
    df["composite_score"] = (
        0.25 * df["rating_score"]
        + 0.20 * df["popularity_score"]
        + 0.12 * df["value_for_money"]
        + 0.12 * df["discount_score"]
        + 0.08 * df["price_score"]
        + 0.06 * df["stock_score"]
        + 0.06 * df["shop_reputation"]
        + 0.05 * df["review_rating_coherence"]
        + 0.03 * df["brand_score"]
        + 0.03 * df["product_completeness"]
    ).clip(0, 1)

    # ── Label Top-K ────────────────────────────────────────────────────────────
    actual_k = min(top_k, n)
    if actual_k > 0 and df["composite_score"].nunique() > 1:
        threshold = df["composite_score"].nlargest(actual_k).min()
        df[TARGET_COLUMN] = (df["composite_score"] >= threshold).astype(int)
    else:
        df[TARGET_COLUMN] = 0

    n_top = df[TARGET_COLUMN].sum()
    logger.success(
        f"[FeatureEng] {len(FEATURE_COLUMNS)} features | "
        f"Top-{actual_k}: {n_top} produits ({n_top/n*100:.1f}%)"
    )
    return df


def get_feature_labels() -> dict[str, str]:
    """Labels lisibles pour le dashboard."""
    return {
        "price_score":              "Prix compétitif",
        "rating_score":             "Note client",
        "popularity_score":         "Popularité (avis)",
        "stock_score":              "Disponibilité stock",
        "discount_score":           "Taux de remise",
        "value_for_money":          "Rapport qualité/prix",
        "price_to_median_ratio":    "Position vs médiane catégorie",
        "log_reviews":              "Log(nb avis)",
        "variant_diversity":        "Richesse variantes",
        "n_images_norm":            "Richesse visuelle",
        "brand_score":              "Score marque",          # NOUVEAU
        "product_completeness":     "Complétude données",
        "has_discount_flag":        "En promotion",
        "is_in_stock_flag":         "En stock",
        "shop_reputation":          "Réputation boutique",   # NOUVEAU
        "review_rating_coherence":  "Cohérence avis/note",   # NOUVEAU
        "title_length_norm":        "Qualité du titre",      # NOUVEAU
        "category_popularity":      "Popularité catégorie",  # NOUVEAU
        "price_volatility":         "Volatilité prix",       # NOUVEAU
        "tag_diversity":            "Diversité mots-clés",   # NOUVEAU
    }


def save_features(df: pd.DataFrame, filename: str = "products_features.csv") -> None:
    out = DATA_PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.success(f"[FeatureEng] → {out}")