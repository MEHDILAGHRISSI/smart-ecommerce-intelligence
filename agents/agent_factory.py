"""Design Pattern Factory pour instancier les agents de scraping."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from agents.base_agent import BaseScraperAgent
from agents.shopify_agent import ShopifyScraperAgent
from agents.woocommerce_agent import WooCommerceScraperAgent


class AgentFactory:
    """
    Fabrique les agents de scraping selon la plateforme demandée.

    Avantage : le code métier n'instancie jamais directement les agents.
    Si on ajoute PrestaShop demain, on touche uniquement cette classe.

    Utilisation :
        agent = AgentFactory.create("shopify", base_url="https://store.myshopify.com")
        products = await agent.scrape()
    """

    _REGISTRY: dict[str, type[BaseScraperAgent]] = {
        "shopify": ShopifyScraperAgent,
        "woocommerce": WooCommerceScraperAgent,
    }

    @classmethod
    def create(
        cls,
        platform: str,
        base_url: str,
        **kwargs,
    ) -> BaseScraperAgent:
        """
        Instancie et retourne l'agent approprié.

        Args:
            platform: Identifiant de la plateforme ("shopify" | "woocommerce")
            base_url: URL de base de la boutique
            **kwargs: Arguments supplémentaires passés au constructeur de l'agent
                      (ex: api_key, consumer_key, consumer_secret)

        Returns:
            Instance de l'agent correspondant (ShopifyScraperAgent ou WooCommerceScraperAgent)

        Raises:
            ValueError: Si la plateforme n'est pas enregistrée
        """
        normalized = platform.lower().strip()
        agent_class = cls._REGISTRY.get(normalized)

        if not agent_class:
            supported = ", ".join(cls._REGISTRY.keys())
            raise ValueError(
                f"Plateforme '{platform}' inconnue. Plateformes supportées : {supported}"
            )

        logger.info(f"[Factory] Création d'un agent '{normalized}' pour {base_url}")
        return agent_class(base_url=base_url, **kwargs)

    @classmethod
    def supported_platforms(cls) -> list[str]:
        """Retourne la liste des plateformes supportées."""
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, platform: str, agent_class: type[BaseScraperAgent]) -> None:
        """
        Enregistre un nouvel agent dans la factory (extension future).

        Args:
            platform: Identifiant unique de la plateforme
            agent_class: Classe héritant de BaseScraperAgent
        """
        cls._REGISTRY[platform.lower()] = agent_class
        logger.info(f"[Factory] Nouvel agent enregistré : '{platform}'")