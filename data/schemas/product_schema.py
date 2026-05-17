"""
data/schemas/product_schema.py — Smart eCommerce Intelligence
==============================================================
Schéma unifié de produit utilisé par tous les agents de scraping.

CORRECTION v2 — Validation robuste des prix :
  Sans @field_validator, Pydantic rejette les valeurs texte fréquentes
  dans le scraping e-commerce réel :
    "1 200,50 MAD"  → ValidationError (rejet du produit)
    "Gratuit"       → ValidationError (rejet du produit)
    "À partir de 15€" → ValidationError (rejet du produit)
    None / ""       → ValidationError (rejet du produit)

  Le validator nettoie ces valeurs AVANT le cast Pydantic,
  garantissant qu'aucun produit ne soit perdu à cause d'un format de prix
  spécifique à une boutique.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _parse_price_string(value: Any) -> float:
    """
    Convertit n'importe quelle représentation de prix en float propre.

    Gère les cas réels rencontrés lors du scraping Shopify / WooCommerce :
    - Déjà numérique : 1200.5  → 1200.5
    - Séparateur espace : "1 200,50 MAD" → 1200.5
    - Virgule décimale : "15,99"  → 15.99
    - Devise collée : "15€", "MAD 250" → 15.0 / 250.0
    - Phrases avec plusieurs nombres : "Lot de 3 t-shirts pour 45.50 MAD" → 45.50 (max)
    - Texte non numérique : "Gratuit", "Free", "À partir de" → 0.0
    - Vide / None → 0.0
    """
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if value is None:
        return 0.0

    raw = str(value).strip()
    if not raw:
        return 0.0

    # Supprime les séparateurs de milliers (espaces et apostrophes)
    # et normalise la virgule décimale → point
    cleaned = raw.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
    cleaned = cleaned.replace(",", ".")

    # Extrait TOUTES les valeurs numériques (flottants) dans la chaîne
    matches = re.findall(r"\d+\.?\d*", cleaned)
    if matches:
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        if numbers:
            # En e-commerce, le prix est généralement le chiffre le plus élevé
            # Exemple : "Lot de 3 t-shirts pour 45.50 MAD" → max(3, 45.50) = 45.50
            return max(0.0, max(numbers))

    return 0.0


class ProductSchema(BaseModel):
    """Représentation canonique d'un produit (Shopify ou WooCommerce)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Données descriptives fondamentales
    id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    brand: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    image_url: Optional[str] = None
    product_url: str

    # Données de prix
    # CORRECTION : mode="before" → _parse_price_string est appelée AVANT
    # que Pydantic ne tente le cast en float.
    # Sans ça, "1 200,50 MAD" lève ValidationError et le produit est perdu.
    price: float = Field(ge=0)
    original_price: Optional[float] = Field(default=None, ge=0)
    currency: str = "MAD"

    # Données de popularité
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    review_count: Optional[int] = Field(default=None, ge=0)

    # Données de stock et logistique
    stock: Optional[int] = Field(default=None, ge=0)
    is_in_stock: bool = True

    # Données sur le vendeur / shop
    shop_name: Optional[str] = None
    shop_country: Optional[str] = None

    # Données sur les variantes (couleurs, tailles, etc.)
    variants: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    # Métadonnées du scraping
    source_platform: str  # "shopify" ou "woocommerce"
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("price", "original_price", mode="before")
    @classmethod
    def clean_price(cls, value: Any) -> float:
        """
        Nettoie toute valeur de prix textuelle avant validation Pydantic.

        Exemples :
            "1 200,50 MAD" → 1200.5
            "Gratuit"      → 0.0
            "À partir de 15€" → 15.0
            None           → 0.0
            150.0          → 150.0  (inchangé)
        """
        return _parse_price_string(value)

    @field_validator("rating", mode="before")
    @classmethod
    def clean_rating(cls, value: Any) -> Optional[float]:
        """
        Nettoie les notes textuelles fréquentes dans le scraping.

        Exemples :
            "4,5/5"     → 4.5
            "4.5 stars" → 4.5
            "N/A"       → None
            None        → None
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            parsed = float(value)
            return parsed if 0.0 <= parsed <= 5.0 else None
        raw = str(value).strip()
        if not raw or raw.lower() in ("n/a", "na", "-", ""):
            return None
        # "4,5/5" ou "4.5 étoiles" → 4.5
        match = re.search(r"\d+[.,]?\d*", raw.replace(",", "."))
        if match:
            try:
                parsed = float(match.group())
                # Normalise les notes sur 10 → sur 5 (WooCommerce)
                if parsed > 5.0:
                    parsed = parsed / 2.0
                return parsed if 0.0 <= parsed <= 5.0 else None
            except ValueError:
                return None
        return None

    @field_validator("review_count", mode="before")
    @classmethod
    def clean_review_count(cls, value: Any) -> Optional[int]:
        """
        Nettoie les compteurs d'avis textuels.

        Exemples :
            "1 234 avis" → 1234
            "(56)"       → 56
            None         → None
        """
        if value is None:
            return None
        if isinstance(value, int):
            return max(0, value)
        raw = str(value).strip()
        match = re.search(r"\d+", raw.replace(" ", "").replace("\u202f", ""))
        if match:
            try:
                return max(0, int(match.group()))
            except ValueError:
                return None
        return None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def discount_percentage(self) -> float:
        """Calcule la remise en % si un ancien prix est disponible."""
        if self.original_price and self.original_price > self.price > 0:
            return round(
                ((self.original_price - self.price) / self.original_price) * 100, 2
            )
        return 0.0