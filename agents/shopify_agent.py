"""Agent de scraping pour les boutiques Shopify (Storefront API + fallback Playwright)."""

from __future__ import annotations

import json
import re
from typing import Optional, Dict, Any, List

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from agents.base_agent import BaseScraperAgent
from agents.exceptions import APIUnavailableError, NormalizationError, ScrapingError
from agents.utils.http_client import HttpClient
from agents.utils.playwright_driver import PlaywrightDriver
from data.schemas.product_schema import ProductSchema


class ShopifyScraperAgent(BaseScraperAgent):
    """
    Agent spécialisé pour les boutiques Shopify.

    Stratégie :
    - Priorité : API JSON native de Shopify (/products.json) — publique, sans clé
    - Fallback  : Playwright scrape la page /collections/all
    - Dans le fallback Playwright, priorité à l'extraction JSON-LD (Schema.org)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        super().__init__(base_url, api_key)

    async def fetch_via_api(self, page: int = 1, per_page: int = 100) -> list[dict]:
        headers = {}
        if self.api_key:
            headers["X-Shopify-Access-Token"] = self.api_key

        try:
            async with HttpClient(self.base_url, headers=headers) as client:
                data = await client.get(
                    "/products.json",
                    params={"limit": per_page, "page": page},
                )
            products = data.get("products", []) if isinstance(data, dict) else []
            logger.debug(f"[shopify] API retourne {len(products)} produits (page {page})")
            return products
        except Exception as e:
            raise APIUnavailableError(f"Erreur API Shopify: {str(e)}") from e

    async def fetch_via_playwright(self, page_num: int = 1) -> list[dict]:
        """
        Scrape une page de collection Shopify avec priorité JSON-LD.
        Retourne une liste de dictionnaires bruts (normalisés ensuite).
        """
        url = f"{self.base_url}/collections/all?page={page_num}"
        try:
            async with PlaywrightDriver(headless=True) as driver:
                page = await driver.new_page()
                await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_selector(".product-item, .product-card, [data-product-id]", timeout=10_000)

                # Récupère le HTML complet pour parser le JSON-LD
                html_content = await page.content()
                products_from_jsonld = self._extract_products_from_jsonld(html_content, self.base_url)

                if products_from_jsonld:
                    logger.info(f"[shopify] JSON-LD trouvé : {len(products_from_jsonld)} produits extraits")
                    return products_from_jsonld

                # Fallback : extraction par sélecteurs CSS (méthode existante)
                logger.warning("[shopify] JSON-LD absent ou vide, fallback vers sélecteurs CSS")
                products = await page.evaluate("""() => {
                    const cards = document.querySelectorAll('.product-item, .product-card, .grid__item');
                    return Array.from(cards).map(card => {
                        const titleEl = card.querySelector('.product-item__title, .product__title, h3, h2');
                        const priceEl = card.querySelector('.price__regular, .product-price, .price');
                        const linkEl = card.querySelector('a[href*="/products/"]');
                        const imgEl = card.querySelector('img');
                        return {
                            title: titleEl?.innerText?.trim() || '',
                            price_raw: priceEl?.innerText?.trim() || '0',
                            product_url: linkEl ? window.location.origin + linkEl.getAttribute('href') : '',
                            image_url: imgEl?.src || '',
                            _source: 'playwright_css_fallback'
                        };
                    }).filter(p => p.title);
                }""")
                return products

        except PlaywrightTimeoutError as e:
            raise ScrapingError(f"Timeout Playwright sur {url} : {e}") from e
        except Exception as e:
            raise ScrapingError(f"Erreur Playwright sur {url}: {str(e)}") from e

    def _extract_products_from_jsonld(self, html: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extrait les produits à partir des balises <script type="application/ld+json">.
        Retourne une liste de dictionnaires conformes au format attendu par normalize().
        """
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", type="application/ld+json")
        products = []

        for script in scripts:
            try:
                data = json.loads(script.string)
                # Gère le cas où data est une liste
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if item.get("@type") == "Product":
                        # Extraction des champs
                        name = item.get("name", "")
                        description = item.get("description", "")
                        offers = item.get("offers", {})
                        price = 0.0
                        currency = "MAD"
                        if offers:
                            price_str = offers.get("price", "0")
                            if isinstance(price_str, str):
                                price_str = price_str.replace(",", ".")
                            try:
                                price = float(price_str)
                            except ValueError:
                                price = 0.0
                            currency = offers.get("priceCurrency", "MAD")

                        # URL du produit (relative ou absolue)
                        product_url = item.get("url", "")
                        if product_url and not product_url.startswith("http"):
                            product_url = base_url.rstrip("/") + "/" + product_url.lstrip("/")

                        # Image principale
                        image = item.get("image", "")
                        if isinstance(image, list) and image:
                            image = image[0]
                        elif isinstance(image, dict):
                            image = image.get("url", "")

                        # SKU ou identifiant (optionnel)
                        sku = item.get("sku", "")

                        product_dict = {
                            "title": name,
                            "description": description,
                            "price_raw": str(price),
                            "currency": currency,
                            "product_url": product_url,
                            "image_url": image,
                            "sku": sku,
                            "_source": "jsonld",
                        }
                        products.append(product_dict)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.debug(f"Erreur parsing JSON-LD : {e}")
                continue

        return products

    def normalize(self, raw: dict) -> ProductSchema:
        try:
            # Cas API : structure Shopify officielle enrichie
            if "variants" in raw or "body_html" in raw:
                # Extraction des tags
                tags_raw = raw.get("tags", [])
                tags = [t.strip() for t in tags_raw.split(",")] if isinstance(tags_raw, str) else tags_raw

                # Extraction des variantes et calcul des prix/stocks globaux
                variants_data = raw.get("variants", [])
                parsed_variants: List[Dict[str, Any]] = []
                price = 0.0
                original_price = None
                total_stock = 0
                is_in_stock = False

                if variants_data:
                    first_variant = variants_data[0]
                    price = float(first_variant.get("price") or 0.0)

                    # Récupération de l'ancien prix (compare_at_price)
                    cmp_price = first_variant.get("compare_at_price")
                    if cmp_price:
                        original_price = float(cmp_price)

                    for v in variants_data:
                        v_stock = v.get("inventory_quantity", 0)
                        total_stock += v_stock
                        if v.get("available") or v_stock > 0:
                            is_in_stock = True

                        parsed_variants.append({
                            "id": str(v.get("id")),
                            "title": v.get("title"),
                            "price": float(v.get("price") or 0.0),
                            "sku": v.get("sku")
                        })
                else:
                    is_in_stock = True  # Présumé en stock si aucune donnée de variante

                image_url = None
                if raw.get("images") and len(raw["images"]) > 0:
                    image_url = raw["images"][0].get("src")

                vendor = raw.get("vendor")

                return ProductSchema(
                    id=str(raw.get("id", "")),
                    title=raw.get("title", "Produit inconnu"),
                    description=_clean_html(raw.get("body_html", "")),
                    category=raw.get("product_type") or None,
                    brand=vendor,
                    tags=tags,
                    image_url=image_url,
                    product_url=f"{self.base_url}/products/{raw.get('handle', '')}",
                    price=price,
                    original_price=original_price,
                    currency="MAD",
                    stock=total_stock if total_stock > 0 else None,
                    is_in_stock=is_in_stock,
                    shop_name=vendor,  # Dans Shopify, le vendeur est souvent le nom du shop
                    variants=parsed_variants,
                    source_platform="shopify"
                )

            # Cas Playwright (CSS ou JSON-LD) : données partielles
            else:
                price = _parse_price(raw.get("price_raw", "0"))
                # Si la devise a été extraite via JSON-LD, on l'utilise
                currency = raw.get("currency", "MAD")
                return ProductSchema(
                    id=_url_to_id(raw.get("product_url", "")),
                    title=raw.get("title", "Produit inconnu"),
                    description=raw.get("description"),
                    price=price,
                    currency=currency,
                    product_url=raw.get("product_url", self.base_url),
                    image_url=raw.get("image_url"),
                    source_platform="shopify_playwright"
                )
        except Exception as e:
            raise NormalizationError(f"Erreur de normalisation: {str(e)}") from e


# --- Helpers privés ---

def _clean_html(html: str) -> str:
    """Supprime les balises HTML d'une description."""
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _parse_price(raw: str) -> float:
    """Extrait le premier nombre décimal trouvé dans une chaîne de prix."""
    match = re.search(r"[\d]+[.,]?[\d]*", raw.replace(",", "."))
    return float(match.group()) if match else 0.0


def _url_to_id(url: str) -> str:
    """Génère un ID depuis le slug de l'URL produit."""
    slug = url.rstrip("/").split("/")[-1]
    return f"shopify_{slug}" if slug else "shopify_unknown"