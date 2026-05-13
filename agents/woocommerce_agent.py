"""Agent de scraping pour les boutiques WooCommerce (REST API v3 + fallback Playwright)."""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

from loguru import logger
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from agents.base_agent import BaseScraperAgent
from agents.exceptions import APIUnavailableError, NormalizationError, ScrapingError
from agents.utils.http_client import HttpClient
from agents.utils.playwright_driver import PlaywrightDriver
from data.schemas.product_schema import ProductSchema


class WooCommerceScraperAgent(BaseScraperAgent):
    """
    Agent spécialisé pour les boutiques WooCommerce.

    Stratégie :
    - Priorité : WooCommerce REST API v3 (/wp-json/wc/v3/products)
    - Fallback  : Playwright scrape la page /shop
    """

    def __init__(
            self,
            base_url: str,
            consumer_key: Optional[str] = None,
            consumer_secret: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, api_key=consumer_key)
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret

    async def fetch_via_api(self, page: int = 1, per_page: int = 100) -> list[dict]:
        import base64

        headers = {}
        if self.consumer_key and self.consumer_secret:
            token = base64.b64encode(
                f"{self.consumer_key}:{self.consumer_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {token}"

        try:
            async with HttpClient(self.base_url, headers=headers) as client:
                data = await client.get(
                    "/wp-json/wc/v3/products",
                    params={"per_page": per_page, "page": page, "status": "publish"},
                )
            products = data if isinstance(data, list) else []
            logger.debug(f"[woocommerce] API retourne {len(products)} produits (page {page})")
            return products
        except Exception as e:
            # Correction: Un seul argument (string)
            raise APIUnavailableError(f"Erreur API WooCommerce: {str(e)}") from e

    async def fetch_via_playwright(self, page_num: int = 1) -> list[dict]:
        url = f"{self.base_url}/shop/page/{page_num}/" if page_num > 1 else f"{self.base_url}/shop/"
        try:
            async with PlaywrightDriver(headless=True) as driver:
                page = await driver.new_page()
                await page.goto(url, wait_until="domcontentloaded")
                await page.wait_for_selector(".product, .wc-block-grid__product", timeout=10_000)

                products = await page.evaluate("""() => {
                    const cards = document.querySelectorAll('.product, .wc-block-grid__product');
                    return Array.from(cards).map(card => {
                        const titleEl = card.querySelector('.woocommerce-loop-product__title, h2');
                        const priceEl = card.querySelector('.price .amount, .woocommerce-Price-amount');
                        const linkEl = card.querySelector('a.woocommerce-loop-product__link, a');
                        const imgEl = card.querySelector('img');
                        const ratingEl = card.querySelector('.star-rating');
                        return {
                            title: titleEl?.innerText?.trim() || '',
                            price_raw: priceEl?.innerText?.trim() || '0',
                            product_url: linkEl?.href || '',
                            image_url: imgEl?.src || '',
                            rating_raw: ratingEl ? ratingEl.getAttribute('aria-label') : null,
                            _source: 'playwright'
                        };
                    }).filter(p => p.title);
                }""")
            return products
        except PlaywrightTimeoutError as e:
            raise ScrapingError(f"Timeout Playwright sur {url}: {e}") from e
        except Exception as e:
            raise ScrapingError(f"Erreur Playwright sur {url}: {str(e)}") from e

    def normalize(self, raw: dict) -> ProductSchema:
        try:
            # Cas API WooCommerce : structure officielle
            if "slug" in raw or "permalink" in raw:
                # Prix courant
                price_str = raw.get("price") or raw.get("regular_price") or "0"
                price = float(price_str) if price_str else 0.0

                # Ancien prix (si en promo)
                original_price = None
                regular_price_str = raw.get("regular_price")
                sale_price_str = raw.get("sale_price")

                if regular_price_str and sale_price_str and float(regular_price_str) > float(sale_price_str):
                    original_price = float(regular_price_str)

                # Stock
                stock = raw.get("stock_quantity") if raw.get("manage_stock") else None
                is_in_stock = raw.get("stock_status") == "instock"

                # Catégories
                categories = raw.get("categories", [])
                category = categories[0].get("name") if categories else None
                sub_category = categories[1].get("name") if len(categories) > 1 else None

                # Tags
                tags_data = raw.get("tags", [])
                tags = [t.get("name") for t in tags_data]

                # Image principale
                images = raw.get("images", [])
                image_url = images[0].get("src") if images else None

                # Note moyenne
                rating_str = raw.get("average_rating", "0")
                rating = float(rating_str) if rating_str else None
                review_count = int(raw.get("rating_count", 0)) or None

                # Nom du shop extrait du base_url
                shop_name = urllib.parse.urlparse(self.base_url).netloc

                return ProductSchema(
                    id=str(raw.get("id", "")),
                    title=raw.get("name", "Produit inconnu"),
                    price=price,
                    original_price=original_price,
                    currency="MAD",  # À adapter selon la boutique
                    rating=rating if rating and rating > 0 else None,
                    review_count=review_count,
                    stock=stock,
                    is_in_stock=is_in_stock,
                    category=category,
                    sub_category=sub_category,
                    tags=tags,
                    shop_name=shop_name,
                    description=_clean_html(raw.get("short_description", "") or raw.get("description", "")),
                    image_url=image_url,
                    product_url=raw.get("permalink", self.base_url),
                    source_platform="woocommerce",
                )

            # Cas Playwright : données partielles
            else:
                price = _parse_price(raw.get("price_raw", "0"))
                rating = _parse_rating(raw.get("rating_raw"))
                shop_name = urllib.parse.urlparse(self.base_url).netloc

                return ProductSchema(
                    id=_url_to_id(raw.get("product_url", "")),
                    title=raw.get("title", "Produit inconnu"),
                    price=price,
                    currency="MAD",
                    rating=rating,
                    shop_name=shop_name,
                    product_url=raw.get("product_url", self.base_url),
                    image_url=raw.get("image_url"),
                    source_platform="woocommerce_playwright",
                )
        except Exception as e:
            raise NormalizationError(f"Erreur de normalisation: {str(e)}") from e


# --- Helpers privés ---

def _clean_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _parse_price(raw: str) -> float:
    match = re.search(r"[\d]+[.,]?[\d]*", raw.replace(",", "."))
    return float(match.group()) if match else 0.0


def _parse_rating(raw: str | None) -> float | None:
    """Extrait la note depuis une chaîne comme 'Rated 4.5 out of 5'."""
    if not raw:
        return None
    match = re.search(r"(\d+\.?\d*)\s*out of\s*(\d+)", raw)
    if match:
        return round(float(match.group(1)) / float(match.group(2)) * 5, 2)
    return None


def _url_to_id(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return f"woo_{slug}" if slug else "woo_unknown"