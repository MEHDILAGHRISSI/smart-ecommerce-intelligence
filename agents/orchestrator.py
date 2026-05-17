"""Orchestrateur Blackboard : échanges Agent-to-Agent via asyncio.Queue."""

from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from agents.agent_factory import AgentFactory
from data.schemas.product_schema import ProductSchema
from llm.description_cleaner import clean_description

OUTPUT_DIR = Path("data/raw")


def _ensure_description(product: ProductSchema) -> ProductSchema:
    """Complète la description manquante via LLM (ou fallback heuristique).

    Robustesse garantie : les erreurs LLM sont loggées mais ne cassent pas le flux.
    """
    if product.description and product.description.strip():
        return product

    # Fallback local pour rester robuste même sans clé LLM.
    fallback = (
        f"{product.title} est un produit de la catégorie "
        f"{product.category or 'Inconnu'}, proposé sur {product.source_platform}."
    )

    try:
        if os.getenv("LLM_API_KEY"):
            enriched_text = clean_description(fallback)
        else:
            enriched_text = fallback
    except Exception as exc:
        logger.warning(f"[Orchestrateur] Erreur LLM pour '{product.title}' : {exc}, utilisation fallback")
        enriched_text = fallback

    return product.model_copy(update={"description": enriched_text})


def _parse_csv_urls(value: str | None) -> list[str]:
    return [url.strip() for url in (value or "").split(",") if url.strip()]


def _build_sources_from_env() -> list[dict[str, Any]]:
    """Construit la liste des boutiques à partir de l'environnement."""
    shopify_raw = (
        os.getenv("SHOPIFY_ALTERNATIVES")
        or os.getenv("SHOPIFY_BASE_URLS")
        or "https://hydrogen-preview.myshopify.com,https://cottonbureau.com"
    )
    woo_raw = (
        os.getenv("WOO_ALTERNATIVES")
        or os.getenv("WOO_BASE_URLS")
        or "https://demo.wpbeaverbuilder.com"
    )

    sources: list[dict[str, Any]] = []
    for url in _parse_csv_urls(shopify_raw):
        sources.append(
            {
                "platform": "shopify",
                "base_url": url,
                "api_key": os.getenv("SHOPIFY_API_KEY") or None,
            }
        )

    for url in _parse_csv_urls(woo_raw):
        sources.append(
            {
                "platform": "woocommerce",
                "base_url": url,
                "consumer_key": os.getenv("WOO_CONSUMER_KEY") or None,
                "consumer_secret": os.getenv("WOO_CONSUMER_SECRET") or None,
            }
        )

    return sources




async def run_agent(platform: str, base_url: str, **kwargs) -> list[ProductSchema]:
    """Lance un agent et retourne ses produits (sans bloquer si erreur)."""
    try:
        max_pages = int(kwargs.pop("max_pages", 5))
        agent = AgentFactory.create(platform, base_url=base_url, **kwargs)
        return await agent.scrape(max_pages=max_pages)
    except Exception as e:
        logger.error(f"[Orchestrateur] Agent '{platform}' échoué : {e}")
        return []


async def _scraper_producer(source: dict[str, Any], queue_raw: asyncio.Queue[dict[str, Any]]) -> None:
    """Agent Scraper : collecte les produits bruts et les dépose dans la file."""
    platform = source["platform"]
    logger.info(f"[Agent Scraper] Démarrage pour {platform} → {source['base_url']}")
    products = await run_agent(
        platform,
        source["base_url"],
        **{k: v for k, v in source.items() if k not in ("platform", "base_url")},
    )

    for product in products:
        await queue_raw.put({"event": "raw_product", "product": product, "platform": platform})

    await queue_raw.put({"event": "producer_done", "platform": platform})
    logger.success(f"[Agent Scraper] {platform} a publié {len(products)} produits")


async def _enricher_consumer(
    queue_raw: asyncio.Queue[dict[str, Any]],
    queue_enriched: asyncio.Queue[dict[str, Any]],
    num_producers: int = 1,
) -> None:
    """Agent Enrichisseur (LLM) : nettoie la description en parallèle du scraping.

    Robustesse : les erreurs lors de _ensure_description sont loggées et le produit continue
    sans description enrichie plutôt que de bloquer l'Event Loop.
    """
    logger.info("[Agent Enrichisseur] Démarrage")
    processed = 0
    producers_done = 0
    while True:
        msg = await queue_raw.get()
        try:
            if msg.get("event") == "producer_done":
                producers_done += 1
                logger.info(f"[Agent Enrichisseur] Producer done ({producers_done}/{num_producers})")
                if producers_done >= num_producers:
                    await queue_enriched.put({"event": "enricher_done"})
                    logger.success(f"[Agent Enrichisseur] Terminé ({processed} produits)")
                    return
                continue

            product: ProductSchema = msg["product"]

            # Enrichissement robuste : capture les erreurs LLM
            try:
                enriched = await asyncio.to_thread(_ensure_description, product)
            except Exception as exc:
                logger.error(f"[Agent Enrichisseur] Enrichissement échoué pour '{product.title}' : {exc}, utilisation produit brut")
                enriched = product

            payload = enriched.model_dump(mode="json")
            payload["discount_pct"] = enriched.discount_percentage
            # Champs par défaut : la détection des anomalies sera faite par le module ML
            payload.setdefault("price_anomaly_flag", 0)
            payload.setdefault("price_anomaly_score", 0.0)

            await queue_enriched.put({"event": "enriched_product", "product": payload})
            processed += 1
        finally:
            queue_raw.task_done()


async def _analyst_consumer(
    queue_enriched: asyncio.Queue[dict[str, Any]],
    sink: list[dict[str, Any]],
) -> None:
    """Agent Analyste (ML) : collecte les produits enrichis dans le sink.

    La détection des anomalies est déléguée au module ML via run_local.py.
    Cet agent accumule simplement les produits enrichis.
    """
    logger.info("[Agent Analyste] Démarrage")
    processed = 0
    while True:
        msg = await queue_enriched.get()
        try:
            if msg.get("event") == "enricher_done":
                logger.success(f"[Agent Analyste] Terminé ({processed} produits)")
                return

            product: dict[str, Any] = msg["product"]
            sink.append(product)
            processed += 1
        finally:
            queue_enriched.task_done()


async def orchestrate(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Orchestre le pattern Blackboard : Scraper -> Enrichisseur -> Analyste.

    ⚠️ IMPORTANT : Cet orchestrateur NE calcule PAS les anomalies.
    La détection des anomalies (price_anomaly_score, etc.) est la responsabilité
    EXCLUSIVE du module ML (ml/clustering.py) via le pipeline de run_local.py.

    Cet orchestrateur se concentre uniquement sur :
    1. Le scraping des produits bruts
    2. L'enrichissement des descriptions via LLM
    3. La collecte des produits enrichis
    """
    if not sources:
        logger.warning("[Blackboard] Aucune source configurée, rien à orchestrer")
        return []

    queue_raw: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
    queue_enriched: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
    sink: list[dict[str, Any]] = []

    producer_tasks = [asyncio.create_task(_scraper_producer(source, queue_raw)) for source in sources]
    num_producers = len(producer_tasks)

    tasks = [*producer_tasks]
    tasks.append(asyncio.create_task(_enricher_consumer(queue_raw, queue_enriched, num_producers)))
    tasks.append(asyncio.create_task(_analyst_consumer(queue_enriched, sink)))

    await asyncio.gather(*tasks)
    logger.success(f"[Blackboard] Pipeline A2A terminé : {len(sink)} produits collectés et enrichis")
    return sink


def save_to_json(products: list[dict[str, Any]], filename: str | None = None) -> Path:
    """Sauvegarde temporaire des produits collectés dans data/raw/ avec timestamp.

    Cette sauvegarde intermédiaire garantit que les données ne sont pas perdues
    en cas d'erreur lors du pipeline ML.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not filename:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"products_{ts}.json"

    path = OUTPUT_DIR / filename
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2, default=str)
        logger.success(f"[Orchestrateur] {len(products)} produits → {path}")
    except Exception as exc:
        logger.error(f"[Orchestrateur] Erreur sauvegarde JSON : {exc}")

    return path


async def main():
    load_dotenv()
    SOURCES = _build_sources_from_env()
    logger.info(f"[Orchestrateur] Sources configurées : {len(SOURCES)} (ex: {SOURCES[:2]})")

    products = await orchestrate(SOURCES)
    if products:
        path = save_to_json(products)
        logger.success(f"✅ {len(products)} produits collectés et enrichis → {path}")
    else:
        logger.warning("⚠️ Aucun produit collecté par le scraping.")


if __name__ == "__main__":
    asyncio.run(main())