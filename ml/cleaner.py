"""Nettoyage et prétraitement des données produits."""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from configs.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR


def load_latest_raw_products(raw_dir: str | Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Charge le fichier JSON le plus récent de data/raw/ → DataFrame."""
    p = Path(raw_dir)
    files = list(p.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier JSON dans {raw_dir}")

    latest = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"[Cleaner] Fichier chargé : {latest.name}")

    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"[Cleaner] {len(df)} produits bruts, {df.shape[1]} colonnes")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame produits."""
    initial = len(df)

    # 1. Déduplication
    dedup_cols = ["id"] if "id" in df.columns else ["title", "source_platform"]
    df = df.drop_duplicates(subset=dedup_cols)
    logger.info(f"[Cleaner] Doublons supprimés : {initial - len(df)}")

    # 2. Titres vides
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]

    # 3. Prix — utilise infer_objects pour éviter FutureWarning pandas
    median_price = df[df["price"] > 0]["price"].median()
    df["price"] = (
        df["price"]
        .replace(0, pd.NA)
        .fillna(median_price)
        .infer_objects(copy=False)
    )

    if "original_price" not in df.columns:
        df["original_price"] = df["price"]
    else:
        df["original_price"] = df["original_price"].fillna(df["price"])

    # 4. Remise
    df["discount_percentage"] = np.where(
        df["original_price"] > df["price"],
        ((df["original_price"] - df["price"]) / df["original_price"] * 100).round(2),
        0.0,
    )

    # 5. Numériques — infer_objects supprime les FutureWarning
    df["rating"] = df["rating"].fillna(0.0).infer_objects(copy=False)
    df["rating"] = df["rating"].clip(0, 5)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["stock"] = df["stock"].fillna(0).astype(int)

    if "is_in_stock" in df.columns:
        df["is_in_stock"] = df["is_in_stock"].fillna(True).astype(bool)

    # 6. Catégorielles
    for col in ["category", "sub_category", "brand", "shop_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu")
    if "category" not in df.columns:
        df["category"] = "Inconnu"

    # 7. Types complexes → string pour CSV
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(
            lambda x: ",".join(x) if isinstance(x, list) else (x or "")
        )
    if "variants" in df.columns:
        df["variants"] = df["variants"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else "[]"
        )

    logger.success(f"[Cleaner] {len(df)} produits après nettoyage")
    return df.reset_index(drop=True)


def save_cleaned(df: pd.DataFrame, filename: str = "products_cleaned.csv") -> Path:
    """Sauvegarde dans data/processed/."""
    out = DATA_PROCESSED_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.success(f"[Cleaner] Sauvegardé → {out}")
    return out