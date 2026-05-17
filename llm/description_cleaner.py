"""
llm/description_cleaner.py — Smart eCommerce Intelligence
==========================================================
Nettoyage et amélioration des descriptions produits via LLM.

Utilise llm.llm_router pour le fallback automatique entre fournisseurs.
"""

from __future__ import annotations
from loguru import logger
from llm.llm_router import call_llm_simple, detect_active_providers


_SYSTEM = (
    "Tu es un expert en rédaction e-commerce. "
    "Tu corriges les descriptions produits : tu retires le jargon promotionnel agressif, "
    "tu corriges les fautes, et tu formates proprement. "
    "Réponds UNIQUEMENT avec la description corrigée, sans explication."
)

_PROMPT_TEMPLATE = (
    "Corrige, nettoie et formate proprement cette description produit e-commerce :\n\n{raw}"
)


def clean_description(raw_description: str) -> str:
    """
    Utilise un LLM pour améliorer une description produit brute.

    Si aucune clé API n'est configurée, retourne la description brute.
    Supporte automatiquement : Groq, Anthropic, OpenAI, Gemini (fallback).
    """
    if not raw_description or not raw_description.strip():
        return raw_description

    # Si aucun fournisseur n'est dispo, on ne gaspille pas de temps
    if not detect_active_providers():
        logger.debug("[DescriptionCleaner] Aucun fournisseur LLM disponible — retour brut.")
        return raw_description

    prompt = _PROMPT_TEMPLATE.format(raw=raw_description[:2000])  # limite de sécurité

    try:
        result = call_llm_simple(prompt, system=_SYSTEM)
        # Vérifier que le résultat est une vraie réponse (pas un message démo)
        if result.startswith("⚠️"):
            return raw_description
        logger.debug(f"[DescriptionCleaner] Description enrichie ({len(result)} chars)")
        return result
    except Exception as exc:
        logger.warning(f"[DescriptionCleaner] Erreur LLM : {exc} — retour brut.")
        return raw_description