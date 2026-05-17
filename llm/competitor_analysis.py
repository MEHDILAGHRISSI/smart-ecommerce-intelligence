"""
llm/competitor_analysis.py — Smart eCommerce Intelligence
==========================================================
Analyse concurrentielle générée par LLM.

Utilise llm.llm_router pour le fallback automatique entre fournisseurs.
"""

from __future__ import annotations
from loguru import logger
from llm.llm_router import call_llm_simple


_SYSTEM = (
    "Tu es un expert en pricing et stratégie e-commerce. "
    "Tu analyses des données produits et concurrents pour donner des recommandations "
    "business claires et actionnables. Réponds en français."
)


def analyze_competitors(product_data: dict, competitors_data: list[dict]) -> str:
    """
    Génère un résumé des forces et faiblesses face à la concurrence.

    Args:
        product_data: Dictionnaire du produit analysé
        competitors_data: Liste de dictionnaires des produits concurrents

    Returns:
        Analyse textuelle avec Avantages, Inconvénients, et Recommandation de prix.
    """
    if not product_data:
        return "Aucune donnée produit fournie pour l'analyse."

    # Construction du prompt structuré
    prod_str = _format_product(product_data)
    comp_str = "\n".join([f"  {i+1}. {_format_product(c)}" for i, c in enumerate(competitors_data[:5])])

    prompt = (
        f"Analyse ce produit e-commerce et ses concurrents :\n\n"
        f"**Produit analysé :**\n{prod_str}\n\n"
        f"**Concurrents (top 5) :**\n{comp_str or '  (aucun concurrent fourni)'}\n\n"
        f"Fournis une analyse structurée en 3 parties :\n"
        f"1. ✅ Avantages concurrentiels\n"
        f"2. ⚠️ Points faibles ou risques\n"
        f"3. 💡 Recommandation de prix et stratégie"
    )

    try:
        result = call_llm_simple(prompt, system=_SYSTEM)
        if result.startswith("⚠️"):
            return "Analyse indisponible — configure une clé API dans .env (GROQ_API_KEY recommandé)."
        return result
    except Exception as exc:
        logger.error(f"[CompetitorAnalysis] Erreur : {exc}")
        return "Erreur lors de la génération de l'analyse concurrentielle."


def _format_product(p: dict) -> str:
    """Formate un produit en string lisible pour le LLM."""
    parts = []
    if p.get("title"):
        parts.append(f"Titre: {p['title'][:60]}")
    if p.get("price") is not None:
        parts.append(f"Prix: {p['price']}")
    if p.get("rating") is not None:
        parts.append(f"Note: {p['rating']}/5")
    if p.get("review_count") is not None:
        parts.append(f"Avis: {p['review_count']}")
    if p.get("category"):
        parts.append(f"Catégorie: {p['category']}")
    if p.get("composite_score") is not None:
        parts.append(f"Score ML: {p['composite_score']:.3f}")
    return " | ".join(parts) if parts else str(p)


def generate_market_report(df_summary: dict) -> str:
    """
    Génère un rapport de marché automatique à partir des statistiques du catalogue.

    Args:
        df_summary: Dictionnaire avec les stats globales (total_products, avg_price, etc.)

    Returns:
        Rapport textuel orienté business.
    """
    prompt = (
        f"Génère un rapport de marché e-commerce concis basé sur ces données :\n\n"
        f"- Produits analysés : {df_summary.get('total_products', 'N/A')}\n"
        f"- Prix moyen : {df_summary.get('avg_price', 'N/A')}\n"
        f"- Note moyenne : {df_summary.get('avg_rating', 'N/A')}/5\n"
        f"- Catégories : {df_summary.get('n_categories', 'N/A')}\n"
        f"- Boutiques : {df_summary.get('n_shops', 'N/A')}\n\n"
        f"Identifie les 3 tendances clés et propose 2 recommandations stratégiques."
    )
    try:
        return call_llm_simple(prompt, system=_SYSTEM)
    except Exception as exc:
        logger.error(f"[CompetitorAnalysis] generate_market_report échoué : {exc}")
        return "Rapport indisponible."