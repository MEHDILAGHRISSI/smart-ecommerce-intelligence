"""Nettoyage et amélioration des descriptions produits via LLM."""

import os
from loguru import logger
import openai


# Remplace par 'anthropic' ou autre selon ton choix de LLM

def clean_description(raw_description: str) -> str:
    """Utilise un LLM pour formater et corriger une description produit."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        logger.error("Clé API LLM introuvable. Retour de la description brute.")
        return raw_description

    client = openai.OpenAI(api_key=api_key)

    prompt = f"Corrige les fautes, retire le jargon promotionnel agressif et formate proprement cette description e-commerce :\n\n{raw_description}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Erreur LLM lors du nettoyage : {e}")
        return raw_description