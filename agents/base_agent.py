"""Classe abstraite de base pour tous les agents de scraping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger

from agents.exceptions import APIUnavailableError, ScrapingError
from data.schemas.product_schema import ProductSchema


class BaseScraperAgent(ABC):
    """
    Contrat abstrait pour tout agent de scraping.

    Implémente la logique API-First avec fallback automatique vers Playwright :
        1. Tente fetch_via_api()
        2. Si APIUnavailableError → bascule sur fetch_via_playwright()
        3. Normalise chaque résultat brut via normalize() → ProductSchema
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        """
        Args:
            base_url: URL de base de la boutique (ex: https://store.myshopify.com)
            api_key: Clé API optionnelle pour authentification
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.platform = self.__class__.__name__.replace("ScraperAgent", "").lower()
        logger.info(f"[{self.platform}] Agent initialisé → {self.base_url}")

    @abstractmethod
    async def fetch_via_api(self, page: int = 1, per_page: int = 100) -> list[dict]:
        """
        Récupère les produits via l'API officielle de la plateforme.

        Args:
            page: Numéro de page pour la pagination
            per_page: Nombre de produits par page

        Returns:
            Liste de produits bruts (dict)

        Raises:
            APIUnavailableError: Si l'API est inaccessible ou retourne une erreur
        """
        ...

    @abstractmethod
    async def fetch_via_playwright(self, page_num: int = 1) -> list[dict]:
        """
        Scrape les produits via le DOM dynamique (fallback Playwright).

        Args:
            page_num: Numéro de page pour la pagination

        Returns:
            Liste de produits bruts extraits du DOM

        Raises:
            ScrapingError: Si le scraping échoue
        """
        ...

    @abstractmethod
    def normalize(self, raw: dict) -> ProductSchema:
        """
        Transforme un produit brut (spécifique à la plateforme) en ProductSchema unifié.

        Args:
            raw: Dictionnaire brut retourné par l'API ou le scraper

        Returns:
            Objet ProductSchema validé par Pydantic

        Raises:
            NormalizationError: Si les données sont trop incomplètes pour être normalisées
        """
        ...

    async def scrape(self, max_pages: int = 10) -> list[ProductSchema]:
        """
        Méthode principale : orchestre la logique API-First + fallback.

        1. Tente de récupérer via l'API officielle (rapide, structuré)
        2. En cas d'échec → bascule sur Playwright (scraping DOM)
        3. Normalise et valide chaque produit avec Pydantic

        Args:
            max_pages: Nombre maximum de pages à scraper

        Returns:
            Liste de ProductSchema validés
        """
        products: list[ProductSchema] = []

        for page_num in range(1, max_pages + 1):
            raw_items: list[dict] = []
            method_used = "api"

            # --- Tentative API ---
            try:
                logger.info(f"[{self.platform}] Tentative API — page {page_num}")
                raw_items = await self.fetch_via_api(page=page_num)
                if not raw_items:
                    logger.info(f"[{self.platform}] API : plus de données à la page {page_num}, arrêt.")
                    break
            except APIUnavailableError as e:
                logger.warning(f"[{self.platform}] API indisponible ({e}), bascule sur Playwright...")
                method_used = "playwright"
                try:
                    raw_items = await self.fetch_via_playwright(page_num=page_num)
                    if not raw_items:
                        logger.info(f"[{self.platform}] Playwright : plus de données à la page {page_num}, arrêt.")
                        break
                except ScrapingError as se:
                    logger.error(f"[{self.platform}] Échec Playwright aussi : {se}. Page ignorée.")
                    continue

            # --- Normalisation ---
            page_products = []
            for raw in raw_items:
                try:
                    product = self.normalize(raw)
                    page_products.append(product)
                except Exception as e:
                    logger.warning(f"[{self.platform}] Normalisation ignorée : {e}")

            products.extend(page_products)
            logger.success(
                f"[{self.platform}] Page {page_num} via {method_used} → "
                f"{len(page_products)} produits | Total: {len(products)}"
            )

        logger.success(f"[{self.platform}] Scraping terminé — {len(products)} produits collectés")
        return products