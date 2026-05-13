"""Règles d'association Apriori entre catégories de produits."""

from __future__ import annotations
import pandas as pd
from loguru import logger
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from configs.settings import MIN_SUPPORT, MIN_CONFIDENCE, DATA_PROCESSED_DIR


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    """
    Construit des 'transactions' à partir des produits scrappés.

    Stratégie : chaque boutique (source_platform + shop_name) représente
    un "panier" contenant toutes ses catégories de produits.

    Cette approche permet de découvrir des co-occurrences de catégories
    fréquemment vendues ensemble par les mêmes marchands.
    """
    group_col = "shop_name" if "shop_name" in df.columns else "source_platform"
    transactions = (
        df.groupby(group_col)["category"]
        .apply(lambda cats: list(cats.dropna().unique()))
        .tolist()
    )
    transactions = [t for t in transactions if len(t) >= 2]  # Au moins 2 catégories
    logger.info(f"[Apriori] {len(transactions)} transactions construites")
    return transactions


def generate_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les règles d'association Apriori.

    Interprétation business :
    - Support élevé : catégories fréquentes dans le catalogue
    - Confiance élevée : si catégorie A → catégorie B très probable
    - Lift > 1 : association non aléatoire (intérêt réel)

    Returns:
        DataFrame de règles triées par lift décroissant,
        ou DataFrame vide si données insuffisantes.
    """
    transactions = build_transactions(df)

    if len(transactions) < 3:
        logger.warning("[Apriori] Moins de 3 transactions. Règles impossibles.")
        return pd.DataFrame()

    te = TransactionEncoder()
    basket_df = pd.DataFrame(
        te.fit_transform(transactions),
        columns=te.columns_,
    )

    frequent = apriori(basket_df, min_support=MIN_SUPPORT, use_colnames=True)
    if frequent.empty:
        logger.warning(f"[Apriori] Aucun itemset fréquent (support>={MIN_SUPPORT}). Essaie de baisser MIN_SUPPORT.")
        return pd.DataFrame()

    rules = association_rules(frequent, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    logger.success(f"[Apriori] {len(rules)} règles générées")
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    logger.info(f"\n{rules[cols].head(10).to_string()}")

    # Sauvegarde
    out = DATA_PROCESSED_DIR / "association_rules.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(out, index=False)
    logger.success(f"[Apriori] Règles → {out}")
    return rules