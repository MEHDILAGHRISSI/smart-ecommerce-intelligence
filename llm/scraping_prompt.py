"""Prompts pour l'extraction de données via LLM."""

EXTRACTION_PROMPT = """
Tu es un expert en extraction de données e-commerce.
Analyse le texte brut suivant provenant d'une page produit et extrais les informations sous format JSON strict.

Le JSON doit respecter cette structure :
{
    "title": "Nom du produit",
    "price": Prix en format numérique (float),
    "description": "Description nettoyée",
    "features": ["caractéristique 1", "caractéristique 2"]
}

Texte brut :
{raw_text}

Renvoie UNIQUEMENT le JSON valide, sans markdown ni texte explicatif.
"""