"""Analyse concurrentielle générée par LLM."""

from loguru import logger
import openai
import os


def analyze_competitors(product_data: dict, competitors_data: list[dict]) -> str:
    """Génère un résumé des forces et faiblesses face à la concurrence."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return "Analyse indisponible (Clé API manquante)."

    client = openai.OpenAI(api_key=api_key)

    prompt = (
        f"En tant qu'analyste e-commerce, compare ce produit : {product_data} "
        f"avec ses concurrents : {competitors_data}. "
        "Dresse un bilan court : Avantages, Inconvénients, et Recommandation de prix."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # ou gpt-3.5-turbo
            messages=[{"role": "system", "content": "Tu es un expert pricing."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur d'analyse concurrentielle : {e}")
        return "Erreur lors de la génération de l'analyse."