"""
Scraping de boutiques Shopify réelles via l'endpoint public /products.json.
Toutes ces boutiques sont publiquement accessibles (pas d'auth requise).

Usage :
    python data/scrape_real_stores.py          # 15 boutiques, ~3000-5000 produits
    python data/scrape_real_stores.py --limit 5  # 5 boutiques seulement
"""

from __future__ import annotations
import asyncio
import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

import httpx
from loguru import logger

OUTPUT_DIR = Path("data/raw")

# ── Boutiques Shopify réelles publiques ───────────────────────────────────────
# Toutes accessibles via /products.json sans authentification
REAL_STORES = [
    # Mode & Vêtements
    {"name": "Allbirds",       "url": "https://www.allbirds.com",       "category_hint": "Fashion"},
    {"name": "Gymshark",       "url": "https://www.gymshark.com",       "category_hint": "Sport"},
    {"name": "Cotopaxi",       "url": "https://www.cotopaxi.com",       "category_hint": "Outdoor"},
    {"name": "Taylor Stitch",  "url": "https://www.taylorstitch.com",   "category_hint": "Fashion"},
    {"name": "Pura Vida",      "url": "https://www.puravidabracelets.com", "category_hint": "Fashion"},

    # Beauté & Cosmétiques
    {"name": "ColourPop",      "url": "https://colourpop.com",          "category_hint": "Beauty"},
    {"name": "Jeffree Star",   "url": "https://jeffreestarcosmetics.com", "category_hint": "Beauty"},
    {"name": "Kylie Cosmetics","url": "https://kyliecosmetics.com",     "category_hint": "Beauty"},

    # Maison & Lifestyle
    {"name": "Brooklinen",     "url": "https://www.brooklinen.com",     "category_hint": "Home"},
    {"name": "Ruggable",       "url": "https://ruggable.com",           "category_hint": "Home"},

    # Alimentation & Boissons
    {"name": "Harney & Sons",  "url": "https://www.harney.com",         "category_hint": "Food"},
    {"name": "Death Wish Coffee", "url": "https://www.deathwishcoffee.com", "category_hint": "Food"},

    # Sport & Outdoor
    {"name": "NoBull",         "url": "https://nobullproject.com",      "category_hint": "Sport"},
    {"name": "Tentree",        "url": "https://www.tentree.com",        "category_hint": "Sport"},

    # Électronique & Tech
    {"name": "Nomad Goods",    "url": "https://www.nomadgoods.com",     "category_hint": "Tech"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SmartEcommerceBot/1.0; FST Tanger Academic Project)",
    "Accept": "application/json",
}


async def fetch_store_products(
    client: httpx.AsyncClient,
    store: dict,
    max_pages: int = 8,
) -> list[dict]:
    """Scrape tous les produits d'une boutique via pagination /products.json."""
    all_products = []
    base_url = store["url"].rstrip("/")

    for page in range(1, max_pages + 1):
        url = f"{base_url}/products.json"
        params = {"limit": 250, "page": page}

        try:
            resp = await client.get(url, params=params, timeout=20.0)
            if resp.status_code == 429:
                logger.warning(f"[{store['name']}] Rate limit. Pause 5s...")
                await asyncio.sleep(5)
                continue
            if resp.status_code != 200:
                logger.warning(f"[{store['name']}] HTTP {resp.status_code} page {page}")
                break

            data = resp.json()
            products = data.get("products", [])

            if not products:
                logger.info(f"[{store['name']}] Plus de produits à la page {page}")
                break

            # Normalisation + enrichissement
            for p in products:
                normalized = _normalize(p, store)
                if normalized:
                    all_products.append(normalized)

            logger.info(f"[{store['name']}] Page {page} → {len(products)} produits")
            await asyncio.sleep(0.3)  # Politesse

        except Exception as e:
            logger.error(f"[{store['name']}] Erreur page {page}: {e}")
            break

    logger.success(f"[{store['name']}] Total : {len(all_products)} produits")
    return all_products


def _normalize(raw: dict, store: dict) -> dict | None:
    """Convertit un produit Shopify brut en format ProductSchema étendu."""
    title = (raw.get("title") or "").strip()
    if not title:
        return None

    variants = raw.get("variants", [])
    first_v = variants[0] if variants else {}

    # Prix
    try:
        price = float(first_v.get("price") or 0)
    except (ValueError, TypeError):
        price = 0.0

    # Ancien prix (compare_at_price = prix barré = indique une promo)
    try:
        original_price = float(first_v.get("compare_at_price") or 0) or None
    except (ValueError, TypeError):
        original_price = None

    # Stock total sur toutes variantes
    total_stock = sum(
        int(v.get("inventory_quantity") or 0)
        for v in variants
        if v.get("inventory_quantity") is not None
    )
    is_in_stock = any(
        v.get("available", False) or (v.get("inventory_quantity") or 0) > 0
        for v in variants
    )

    # Catégorie : product_type > tags > category_hint de la boutique
    product_type = (raw.get("product_type") or "").strip()
    tags = raw.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    category = product_type or (tags[0] if tags else store.get("category_hint", "Inconnu"))

    # Image
    images = raw.get("images", [])
    image_url = images[0].get("src") if images else None

    # Remise
    discount_pct = 0.0
    if original_price and original_price > price > 0:
        discount_pct = round((original_price - price) / original_price * 100, 2)

    # Nombre de variantes (feature utile)
    n_variants = len(variants)
    n_images = len(images)

    return {
        "id": f"shopify_{store['name'].replace(' ', '_').lower()}_{raw.get('id', '')}",
        "title": title,
        "price": price,
        "original_price": original_price,
        "currency": "USD",
        "rating": None,          # Shopify API publique ne fournit pas les notes
        "review_count": None,
        "stock": total_stock if total_stock > 0 else None,
        "is_in_stock": is_in_stock,
        "category": category,
        "brand": raw.get("vendor") or store["name"],
        "shop_name": store["name"],
        "tags": tags[:5],        # 5 premiers tags seulement
        "n_variants": n_variants,
        "n_images": n_images,
        "discount_pct": discount_pct,
        "has_discount": discount_pct > 0,
        "description": None,     # body_html volontairement ignoré (trop lourd)
        "image_url": image_url,
        "product_url": f"{store['url'].rstrip('/')}/products/{raw.get('handle', '')}",
        "source_platform": "shopify",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


async def scrape_all(stores: list[dict], max_pages: int = 8) -> list[dict]:
    """Scrape toutes les boutiques en parallèle (concurrence limitée à 3)."""
    semaphore = asyncio.Semaphore(3)  # Max 3 boutiques simultanées

    async def bounded_scrape(store):
        async with semaphore:
            async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
                return await fetch_store_products(client, store, max_pages)

    results = await asyncio.gather(*[bounded_scrape(s) for s in stores])

    all_products = [p for batch in results for p in batch]
    logger.success(f"Total scraping : {len(all_products)} produits depuis {len(stores)} boutiques")
    return all_products


def save(products: list[dict]) -> Path:
    """Sauvegarde dans data/raw/ avec timestamp."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"products_{ts}_real_shopify.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2, default=str)
    logger.success(f"Sauvegardé → {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(REAL_STORES),
                        help="Nombre de boutiques à scraper")
    parser.add_argument("--pages", type=int, default=8,
                        help="Pages max par boutique (250 produits/page)")
    args = parser.parse_args()

    stores_to_scrape = REAL_STORES[:args.limit]
    logger.info(f"Scraping de {len(stores_to_scrape)} boutiques réelles Shopify...")
    logger.info(f"Boutiques : {[s['name'] for s in stores_to_scrape]}")

    products = asyncio.run(scrape_all(stores_to_scrape, max_pages=args.pages))

    if products:
        path = save(products)
        print(f"\n✅ {len(products)} produits réels → {path}")
        # Stats par boutique
        from collections import Counter
        by_shop = Counter(p["shop_name"] for p in products)
        print("\nProduits par boutique :")
        for shop, count in sorted(by_shop.items(), key=lambda x: -x[1]):
            print(f"  {shop:<25} {count:>5} produits")
    else:
        print("\n⚠️ Aucun produit. Vérifie ta connexion internet.")
        print("Certaines boutiques peuvent bloquer les bots selon les régions.")
        print("Alternative : python data/generate_synthetic.py 2000")