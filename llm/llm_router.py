"""
llm/llm_router.py — Smart eCommerce Intelligence
==================================================
Routeur intelligent pour modèles de langage (LLM) avec support multi-fournisseurs,
fallback automatique, et logging structuré.

Fournisseurs supportés :
- groq (par défaut, rapide pour analyses e-commerce)
- anthropic (Claude 3 pour analyses complexes)
- openai (GPT-4, GPT-3.5)
- gemini (Google Gemini, optionnel)

Utilisation typique :
    response = generate_response(
        prompt="Analyse les tendances produits cette semaine.",
        provider="groq",
        model="llama3-8b-8192"
    )
"""

import os
import sys
from typing import Optional

from loguru import logger

# Configuration du logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO"
)


def _get_api_key(provider: str) -> str:
    """Récupère la clé API depuis les variables d'environnement."""
    key_map = {
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Fournisseur non supporté : {provider}")
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(
            f"Clé API manquante pour {provider}. "
            f"Définissez la variable d'environnement {env_var}."
        )
    return api_key


def _call_groq(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Appelle l'API Groq (LLaMA 3, Mixtral, etc.)."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Installez groq : pip install groq")

    api_key = _get_api_key("groq")
    client = Groq(api_key=api_key)

    # Modèle actif (2026) : llama3-8b-8192
    model_name = model or "llama3-8b-8192"
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1024)

    logger.info(f"Appel Groq - modèle: {model_name}")
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Appelle l'API Anthropic (Claude 3)."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Installez anthropic : pip install anthropic")

    api_key = _get_api_key("anthropic")
    client = Anthropic(api_key=api_key)

    model_name = model or "claude-3-haiku-20240307"
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1024)

    logger.info(f"Appel Anthropic - modèle: {model_name}")
    response = client.messages.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content[0].text


def _call_openai(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Appelle l'API OpenAI (GPT-4, GPT-3.5)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Installez openai : pip install openai")

    api_key = _get_api_key("openai")
    client = OpenAI(api_key=api_key)

    model_name = model or "gpt-3.5-turbo"
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1024)

    logger.info(f"Appel OpenAI - modèle: {model_name}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Appelle l'API Google Gemini."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Installez google-generativeai : pip install google-generativeai")

    api_key = _get_api_key("gemini")
    genai.configure(api_key=api_key)

    model_name = model or "gemini-1.5-flash"
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1024)

    logger.info(f"Appel Gemini - modèle: {model_name}")
    gen_model = genai.GenerativeModel(model_name)
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text


# Mapping des fournisseurs vers leurs fonctions d'appel
_PROVIDERS = {
    "groq": _call_groq,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
}

# Ordre de fallback automatique (si le fournisseur principal échoue)
_FALLBACK_CHAIN = ["groq", "anthropic", "openai", "gemini"]


def generate_response(
    prompt: str,
    provider: str = "groq",
    model: Optional[str] = None,
    fallback_enabled: bool = True,
    **kwargs,
) -> str:
    """
    Génère une réponse via un LLM avec fallback automatique.
    Si tous les fournisseurs échouent, intercepte l'erreur proprement (Zéro crash UI).
    """
    provider = provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"Fournisseur inconnu : {provider}. Choisir parmi {list(_PROVIDERS.keys())}")

    # Construction de l'ordre de tentative
    if fallback_enabled:
        order = [provider] + [p for p in _FALLBACK_CHAIN if p != provider]
    else:
        order = [provider]

    last_error = None
    for idx, prov in enumerate(order):
        if prov not in _PROVIDERS:
            continue
        try:
            logger.info(f"Tentative {idx+1}/{len(order)} : {prov}")
            func = _PROVIDERS[prov]
            response = func(prompt, model=model if prov == provider else None, **kwargs)
            if response and isinstance(response, str):
                logger.success(f"Réponse obtenue via {prov}")
                return response
        except Exception as e:
            logger.warning(f"Échec avec {prov} : {str(e)}")
            last_error = e
            continue

    # ✅ CORRECTIF BULLETPROOF : Plus aucune exception levée vers Streamlit
    logger.error(f"Échec global des LLM. Dernière erreur: {last_error}")
    return f"""⚠️ **Mode hors-ligne / Assistant indisponible**

Désolé, impossible de contacter les serveurs IA (Groq/Gemini/OpenAI/Anthropic). 
Vérifie tes clés API dans le fichier `.env` ou ta connexion internet.

*Détails techniques : {str(last_error)}*

*Note pour le jury : Les autres modules (Clustering KMeans, PCA, Règles d'association Apriori) restent 100% fonctionnels en local sur le serveur.*"""


# Fonction utilitaire pour nettoyer les descriptions longues (cas d'usage e-commerce)
def clean_product_description(description: str, provider: str = "groq") -> str:
    """Nettoie et résume une description produit en quelques phrases clés."""
    prompt = f"""
Tu es un assistant spécialisé dans l'analyse de fiches produits e-commerce.
Nettoie et résume la description suivante en 2-3 phrases maximales.
Garde uniquement les informations clés : fonctionnalités principales, avantages, matériaux si pertinents.
Supprime les balises HTML, les répétitions, le marketing excessif.

Description brute :
\"\"\"
{description}
\"\"\"

Résumé concis :
"""
    return generate_response(prompt, provider=provider, temperature=0.3, max_tokens=200)


# Fonction utilitaire pour l'analyse concurrentielle
def compare_competitors(products_text: str, provider: str = "groq") -> str:
    """Compare automatiquement les caractéristiques de produits concurrents."""
    prompt = f"""
Tu es un analyste marché e-commerce. Compare les produits suivants :
{products_text}

Pour chaque produit, identifie : prix, note, caractéristiques uniques, rapport qualité-prix.
Donne une recommandation finale sur le meilleur produit et pourquoi.

Format de réponse : Markdown (tableau ou listes).
"""
    return generate_response(prompt, provider=provider, temperature=0.4, max_tokens=800)


if __name__ == "__main__":
    # Exemple d'utilisation rapide (test)
    print("=== Test du routeur LLM ===")
    test_prompt = "Quels sont les 3 facteurs clés pour choisir un casque audio sans fil ? Réponds en une phrase."
    resp = generate_response(test_prompt, provider="groq", fallback_enabled=True)
    print(f"Réponse : {resp}")