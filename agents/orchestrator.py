"""Orchestrateur : lance les agents en parallèle et sauvegarde les résultats."""

from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from agents.agent_factory import AgentFactory
from data.schemas.product_schema import ProductSchema

OUTPUT_DIR = Path("data/raw")


async def run_agent(platform: str, base_url: str, **kwargs) -> list[ProductSchema]:
    """Lance un agent et retourne ses produits (sans bloquer si erreur)."""
    try:
        agent = AgentFactory.create(platform, base_url=base_url, **kwargs)
        return await agent.scrape(max_pages=5)
    except Exception as e:
        logger.error(f"[Orchestrateur] Agent '{platform}' échoué : {e}")
        return []


async def orchestrate(sources: list[dict]) -> list[ProductSchema]:
    """
    Lance tous les agents en parallèle.

    Args:
        sources: Liste de configs :
            [{"platform": "shopify", "base_url": "https://..."},
             {"platform": "woocommerce", "base_url": "https://...", ...}]
    """
    logger.info(f"[Orchestrateur] {len(sources)} agents en parallèle...")

    tasks = [
        run_agent(
            s["platform"], s["base_url"],
            **{k: v for k, v in s.items() if k not in ("platform", "base_url")},
        )
        for s in sources
    ]

    all_products: list[ProductSchema] = []
    for batch in await asyncio.gather(*tasks):
        all_products.extend(batch)

    logger.success(f"[Orchestrateur] Total : {len(all_products)} produits")
    return all_products


def save_to_json(products: list[ProductSchema], filename: str | None = None) -> Path:
    """Sauvegarde dans data/raw/ avec timestamp."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"products_{ts}.json"

    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [p.model_dump(mode="json") for p in products],
            f, ensure_ascii=False, indent=2, default=str,
        )
    logger.success(f"[Orchestrateur] {len(products)} produits → {path}")
    return path


async def main():
    load_dotenv()

    SOURCES = [
        {
            "platform": "shopify",
            "base_url": os.getenv("SHOPIFY_BASE_URL", "https://hydrogen-preview.myshopify.com"),
            "api_key": os.getenv("SHOPIFY_API_KEY") or None,
        },
        {
            "platform": "woocommerce",
            "base_url": os.getenv("WOO_BASE_URL", "https://demo.wpbeaverbuilder.com"),
            "consumer_key": os.getenv("WOO_CONSUMER_KEY") or None,
            "consumer_secret": os.getenv("WOO_CONSUMER_SECRET") or None,
        },
    ]

    products = await orchestrate(SOURCES)
    if products:
        path = save_to_json(products)
        print(f"\n✅ {len(products)} produits → {path}")

        # Conseil si données insuffisantes pour le ML
        if len(products) < 50:
            print(f"\n⚠️  Seulement {len(products)} produits collectés.")
            print("   Le pipeline ML nécessite au moins 50 produits pour la classification.")
            print("   Génère des données synthétiques pour compléter :")
            print("   python data/generate_synthetic.py 500")
    else:
        print("\n⚠️  Aucun produit collecté.")
        print("Les URLs de démo sont peut-être indisponibles.")
        print("Génère des données synthétiques :")
        print("   python data/generate_synthetic.py 500")


if __name__ == "__main__":
    asyncio.run(main())