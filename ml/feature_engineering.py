"""
Feature Engineering étendu — 20 features explicatives pour le pipeline ML.
Ce module ne crée pas de cible supervisée : aucune variable de type
``is_top_product`` n'est dérivée ici.

Groupes :
  Groupe 1 — Scores normalisés [0,1]     : 5 features
  Groupe 2 — Features dérivées           : 6 features (+ brand_score)
  Groupe 3 — Features métier             : 5 features (+ shop_reputation, review_rating_coherence)
  Groupe 4 — Features texte/diversité   : 4 features (title_length, category_popularity,
                                                        price_volatility, tag_diversity)

CORRECTION v2 :
  - price_score : normalisation par quantile 95% au lieu du max()
    → robuste aux prix aberrants (un outlier à 50 000 MAD n'écrase plus le reste)
  - stock_score, discount_score : même correction quantile 95%
"""
from __future__ import annotations
from collections.abc import Iterable, Sequence
import numpy as np
import pandas as pd
from loguru import logger
from configs.settings import DATA_PROCESSED_DIR

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
    "brand_score",
    # Groupe 3 : Features métier
    "product_completeness",
    "has_discount_flag",
    "is_in_stock_flag",
    "shop_reputation",
    "review_rating_coherence",
    # Groupe 4 : Features texte/diversité
    "title_length_norm",
    "category_popularity",
    "price_volatility",
    "tag_diversity",
]

TARGET_COLUMN = "is_top_product"
TARGET_COLUMNS = ("is_top_product", "is_success")
DEFAULT_LEAKAGE_COLUMNS = ("cluster", "cluster_label", "composite_score")


def resolve_target_column(df: pd.DataFrame, preferred_targets: Sequence[str] = TARGET_COLUMNS) -> str:
    """Retourne la première cible disponible dans ``df``."""
    for column in preferred_targets:
        if column in df.columns:
            return column
    raise KeyError(f"Aucune cible trouvée parmi {list(preferred_targets)}")


def select_model_features(
    df: pd.DataFrame,
    *,
    candidate_columns: Sequence[str] | None = None,
    target_column: str | None = None,
    leakage_columns: Iterable[str] | None = None,
) -> list[str]:
    """Sélectionne les colonnes de X en excluant cible et variables fuite."""
    candidates = list(candidate_columns or FEATURE_COLUMNS)
    forbidden: set[str] = {str(column) for column in DEFAULT_LEAKAGE_COLUMNS}
    if target_column:
        forbidden.add(str(target_column))
    forbidden.update(str(column) for column in TARGET_COLUMNS)
    if leakage_columns:
        forbidden.update(str(column) for column in leakage_columns)
    return [col for col in candidates if col in df.columns and col not in forbidden]


def _robust_normalize(series: pd.Series, quantile: float = 0.95) -> pd.Series:
    """
    Normalisation robuste par quantile [0, 1].

    Avantage vs max() :
    Un seul outlier extrême (ex: produit de luxe à 50 000 MAD) ne compresse
    plus tous les autres produits vers 0. Le quantile 95% représente
    la valeur typique haute du catalogue.

    Tout ce qui dépasse le quantile est clipé à 1.0.
    """
    q = series.quantile(quantile)
    if q > 0:
        return (series / q).clip(0, 1)
    return pd.Series(0.5, index=series.index)


def add_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit 20 features ML sans cible supervisée.

    Corrections v2 :
    - price_score, stock_score, discount_score normalisés par quantile 95%
      (robuste aux outliers de prix).
    """
    df = df.copy()

    # ── Groupe 1 : Scores normalisés [0,1] ───────────────────────────────────

    # CORRIGÉ : quantile 95% au lieu de max() → robuste aux prix aberrants
    df["price_score"] = _robust_normalize(df["price"], quantile=0.95).apply(
        lambda x: 1 - x  # prix bas = score haut
    ).clip(0, 1)

    df["rating_score"] = (df["rating"] / 5.0).clip(0, 1)

    df["log_reviews"] = np.log1p(df["review_count"])
    max_log = df["log_reviews"].max()
    df["popularity_score"] = (df["log_reviews"] / max_log).clip(0, 1) if max_log > 0 else 0.0

    # CORRIGÉ : quantile 95% pour le stock
    df["stock_score"] = _robust_normalize(df["stock"], quantile=0.95).clip(0, 1)

    disc_col = "discount_pct" if "discount_pct" in df.columns else "discount_percentage"
    if disc_col in df.columns:
        # CORRIGÉ : quantile 95% pour les remises
        df["discount_score"] = _robust_normalize(df[disc_col], quantile=0.95).clip(0, 1)
    else:
        df["discount_score"] = 0.0

    # ── Groupe 2 : Features dérivées ─────────────────────────────────────────
    df["value_for_money"] = (df["rating_score"] * df["price_score"]).clip(0, 1)
    df.loc[df["rating"] == 0, "value_for_money"] = 0.0

    cat_medians = df.groupby("category")["price"].transform("median")
    ratio = (df["price"] / cat_medians.replace(0, 1)).clip(0, 5)
    df["price_to_median_ratio"] = (1 - (ratio - 1).abs().clip(0, 4) / 4)

    if "n_variants" in df.columns:
        df["variant_diversity"] = _robust_normalize(
            df["n_variants"].clip(upper=20), quantile=0.95
        ).clip(0, 1)
    else:
        df["variant_diversity"] = 0.0

    if "n_images" in df.columns:
        df["n_images_norm"] = _robust_normalize(
            df["n_images"].clip(upper=10), quantile=0.95
        ).clip(0, 1)
    else:
        df["n_images_norm"] = 0.5

    df["log_price"] = pd.Series(
        np.log1p(pd.to_numeric(df["price"], errors="coerce").clip(lower=0).fillna(0.0)),
        index=df.index,
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # brand_score : fréquence relative de la marque dans le catalogue
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

    # shop_reputation : note moyenne des produits du même shop
    if "shop_name" in df.columns:
        shop_avg_rating = df.groupby("shop_name")["rating"].transform("mean")
        df["shop_reputation"] = (shop_avg_rating / 5.0).clip(0, 1)
    else:
        df["shop_reputation"] = 0.5

    # review_rating_coherence : produits populaires ET bien notés
    df["review_rating_coherence"] = (df["popularity_score"] * df["rating_score"]).clip(0, 1)

    # ── Groupe 4 : Features texte / diversité ─────────────────────────────────
    # title_length_norm : titre ni trop court ni trop long → proxy de qualité
    if "title" in df.columns:
        title_len = df["title"].str.len().fillna(0)
        df["title_length_norm"] = ((title_len.clip(10, 100) - 10) / 90).clip(0, 1)
    else:
        df["title_length_norm"] = 0.5

    # category_popularity : fréquence de la catégorie = proxy de marché actif
    cat_counts = df["category"].value_counts(normalize=True)
    df["category_popularity"] = df["category"].map(cat_counts).fillna(0.0).clip(0, 1)

    # price_volatility : écart relatif au prix médian de la catégorie
    df["price_volatility"] = (ratio - 1.0).abs().clip(0, 4) / 4

    # tag_diversity : richesse des mots-clés du produit
    if "tags" in df.columns:
        tag_counts_series = df["tags"].fillna("").apply(
            lambda x: len(str(x).split(",")) if x and str(x) != "nan" else 0
        )
        max_tags = tag_counts_series.clip(upper=10).max()
        df["tag_diversity"] = (tag_counts_series.clip(upper=10) / max_tags).clip(0, 1) if max_tags > 0 else 0.0
    else:
        df["tag_diversity"] = 0.0

    # ── Score composite pondéré (ranking uniquement — JAMAIS cible de modèle) ─
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

    logger.success(
        f"[FeatureEng] {len(FEATURE_COLUMNS)} features construites "
        "(normalisation quantile 95% — robuste aux outliers de prix)"
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
        "brand_score":              "Score marque",
        "product_completeness":     "Complétude données",
        "has_discount_flag":        "En promotion",
        "is_in_stock_flag":         "En stock",
        "shop_reputation":          "Réputation boutique",
        "review_rating_coherence":  "Cohérence avis/note",
        "title_length_norm":        "Qualité du titre",
        "category_popularity":      "Popularité catégorie",
        "price_volatility":         "Volatilité prix",
        "tag_diversity":            "Diversité mots-clés",
    }


def save_features(df: pd.DataFrame, filename: str = "products_features.csv") -> None:
    out = DATA_PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.success(f"[FeatureEng] → {out}")