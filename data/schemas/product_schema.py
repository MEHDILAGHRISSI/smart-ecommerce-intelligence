"""Schéma unifié de produit utilisé par tous les agents de scraping."""

from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Any, Optional


class ProductSchema(BaseModel):
    """Représentation canonique d'un produit (Shopify ou WooCommerce)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Données descriptives fondamentales
    id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None # Ajouté pour une meilleure segmentation
    brand: Optional[str] = None # Ajouté : Marque ou vendeur
    tags: Optional[List[str]] = Field(default_factory=list) # Ajouté : Mots-clés
    image_url: Optional[str] = None
    product_url: str

    # Données de prix
    price: float = Field(ge=0)
    original_price: Optional[float] = Field(default=None, ge=0) # Ajouté : Ancien prix
    currency: str = "MAD"

    # Données de popularité
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    review_count: Optional[int] = Field(default=None, ge=0)

    # Données de stock et logistique
    stock: Optional[int] = Field(default=None, ge=0)
    is_in_stock: bool = True # Ajouté : Disponibilité booléenne

    # Données sur le vendeur / shop
    shop_name: Optional[str] = None # Ajouté : Nom du shop
    shop_country: Optional[str] = None # Ajouté : Pays du shop

    # Données sur les variantes (couleurs, tailles, etc.)
    variants: Optional[List[Dict[str, Any]]] = Field(default_factory=list) # Ajouté

    # Métadonnées du scraping
    source_platform: str  # "shopify" ou "woocommerce"
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def discount_percentage(self) -> float:
        """Calcule la remise si un ancien prix est disponible."""
        if self.original_price and self.original_price > self.price:
            return round(((self.original_price - self.price) / self.original_price) * 100, 2)
        return 0.0