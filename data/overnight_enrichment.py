"""
🌙 Overnight Enrichment Script
================================
Script robuste pour enrichir des milliers de produits Shopify pendant la nuit.

Fonctionnalités :
    - ✅ Checkpointing : sauvegarde tous les BATCH_SAVE_SIZE produits
    - ✅ Reprise sur erreur : relance le script → continue là où ça s'est arrêté
    - ✅ Gestion RAM : Semaphore(3) + fermeture stricte page/context
    - ✅ Anti-ban : délais aléatoires entre les batchs
    - ✅ Logging détaillé avec progression

Usage :
    python data/overnight_enrichment.py
    python data/overnight_enrichment.py --input data/raw/latest_products.json
    python data/overnight_enrichment.py --input data/raw/latest_products.json --batch 50 --concurrency 3
"""

import asyncio
import json
import os
import random
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright
from loguru import logger

# ─────────────────────────────────────────────
# Configuration par défaut
# ─────────────────────────────────────────────
DEFAULT_INPUT  = "data/raw/latest_products.json"
DEFAULT_OUTPUT = "data/raw/products_enriched_overnight.json"
BATCH_SAVE_SIZE = 50      # Sauvegarder tous les N produits
CONCURRENCY     = 3       # Navigateurs parallèles (3 = safe pour la RAM)
PAGE_TIMEOUT    = 60_000  # 60s max par page
WAIT_JS         = 3_000   # 3s pour laisser charger le JS tiers (Judge.me, Yotpo…)


# ─────────────────────────────────────────────
# Extraction des données (logique identique à enrich_products.py)
# ─────────────────────────────────────────────
async def extract_product_data(page, url: str) -> dict:
    """
    Visite une page produit et extrait rating, review_count, stock.
    Retourne un dict avec les champs enrichis (None si échec).
    """
    try:
        await page.goto(url, wait_until="load", timeout=PAGE_TIMEOUT)

        # Pause pour laisser les apps tierces (Judge.me, Yotpo…) injecter leur JSON-LD
        await page.wait_for_timeout(WAIT_JS)

        # Scroll léger : déclenche le lazy-loading des widgets d'avis
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(1_000)

        # ── 1. Rating & Review count via JSON-LD (méthode principale) ──────────
        rating_data = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script[type="application/ld+json"]');
            for (let script of scripts) {
                try {
                    const data = JSON.parse(script.innerText);
                    const items = Array.isArray(data) ? data : [data];
                    for (let item of items) {
                        if (item['@type'] === 'Product' && item.aggregateRating) {
                            return {
                                rating: parseFloat(item.aggregateRating.ratingValue),
                                count:  parseInt(item.aggregateRating.reviewCount)
                            };
                        }
                    }
                } catch(e) {}
            }
            return null;
        }""")

        # Fallback CSS si JSON-LD absent (Judge.me, Loox, Stamped…)
        if rating_data is None:
            rating_css = await page.evaluate("""() => {
                const el = document.querySelector(
                    '.jdgm-prev-badge__avg-rating, ' +
                    '.spr-badge-caption, ' +
                    '.okeReviews-starRating--star, ' +
                    '[data-rating], ' +
                    '.stamped-badge-caption'
                );
                if (!el) return null;
                const m = el.innerText.match(/(\\d+\\.?\\d*)/);
                return m ? { rating: parseFloat(m[0]), count: null } : null;
            }""")
            rating_data = rating_css

        rating       = rating_data["rating"] if rating_data else None
        review_count = rating_data["count"]  if rating_data else None

        # ── 2. Stock via bouton "Add to cart" ──────────────────────────────────
        is_instock = await page.evaluate("""() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            const addBtn = buttons.find(b =>
                b.innerText.toLowerCase().includes('add') ||
                b.innerText.toLowerCase().includes('ajouter')
            );
            return addBtn ? !addBtn.disabled : true;
        }""")

        # ── 3. Stock exact via Cart Hack AJAX (Shopify natif) ──────────────────
        exact_stock = await page.evaluate("""async () => {
            try {
                const idInput = document.querySelector(
                    'form[action^="/cart/add"] [name="id"], select[name="id"]'
                );
                if (!idInput) return null;
                const variantId = idInput.value;
                const res = await fetch('/cart/add.js', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ items: [{ id: variantId, quantity: 9999 }] })
                });
                const data = await res.json();
                if (res.status === 422 && data.description) {
                    const match = data.description.match(/\\d+/);
                    return match ? parseInt(match[0]) : 0;
                }
                if (res.ok) {
                    await fetch('/cart/clear.js', { method: 'POST' });
                    return 9999;
                }
            } catch(e) { return null; }
            return null;
        }""")

        return {
            "rating":       rating,
            "review_count": review_count,
            "is_in_stock":  is_instock,
            "stock":        exact_stock if exact_stock is not None else (1 if is_instock else 0),
            "enriched_at":  datetime.now(timezone.utc).isoformat(),
            "_enrichment_status": "success",
        }

    except Exception as e:
        logger.warning(f"⚠️  Erreur sur {url[:60]}… → {str(e)[:80]}")
        return {
            "rating":       None,
            "review_count": None,
            "is_in_stock":  True,
            "stock":        None,
            "enriched_at":  datetime.now(timezone.utc).isoformat(),
            "_enrichment_status": "error",
        }


# ─────────────────────────────────────────────
# Traitement d'un produit (avec gestion RAM stricte)
# ─────────────────────────────────────────────
async def process_product(browser, product: dict, semaphore: asyncio.Semaphore) -> dict:
    """
    Ouvre un contexte isolé par produit → visite la page → ferme proprement.
    Le Semaphore limite le nombre de contextes ouverts simultanément.
    """
    async with semaphore:
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        try:
            url = product.get("product_url", "")
            if not url:
                logger.warning(f"Pas d'URL pour le produit {product.get('id', '?')}")
                return product

            enrichment = await extract_product_data(page, url)
            product.update(enrichment)

            status_icon = "✅" if enrichment["_enrichment_status"] == "success" else "⚠️ "
            logger.info(
                f"{status_icon} {product.get('title', '?')[:40]:40s} "
                f"| ⭐ {enrichment['rating']} "
                f"({enrichment['review_count']} avis) "
                f"| 📦 stock={enrichment['stock']}"
            )
        finally:
            # Fermeture OBLIGATOIRE pour libérer la RAM (Playwright garde les contextes en mémoire)
            await page.close()
            await context.close()

        return product


# ─────────────────────────────────────────────
# Gestion du checkpoint (sauvegarde / reprise)
# ─────────────────────────────────────────────
def load_checkpoint(output_file: str) -> tuple[list, set]:
    """Charge les produits déjà enrichis et retourne leurs IDs."""
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            done = json.load(f)
        done_ids = {p["id"] for p in done if "id" in p}
        logger.info(f"📂 Checkpoint trouvé : {len(done)} produits déjà enrichis.")
        return done, done_ids
    except (json.JSONDecodeError, KeyError):
        logger.warning("⚠️  Checkpoint corrompu, on repart de zéro.")
        return [], set()


def save_checkpoint(data: list, output_file: str) -> None:
    """Sauvegarde atomique : écrit d'abord dans un fichier .tmp puis renomme."""
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_file, output_file)  # Atomique : évite un fichier corrompu en cas de crash


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
async def main(input_file: str, output_file: str, batch_size: int, concurrency: int) -> None:
    logger.info("=" * 60)
    logger.info("🌙  OVERNIGHT ENRICHMENT SCRAPER — Démarrage")
    logger.info(f"   Input  : {input_file}")
    logger.info(f"   Output : {output_file}")
    logger.info(f"   Batch  : {batch_size} produits | Concurrency : {concurrency}")
    logger.info("=" * 60)

    # ── Charger les produits bruts ──────────────────────────────────────────
    if not os.path.exists(input_file):
        logger.error(f"❌ Fichier introuvable : {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        all_products = json.load(f)
    logger.info(f"📥 {len(all_products)} produits chargés depuis {input_file}")

    # ── Charger le checkpoint (produits déjà faits) ─────────────────────────
    enriched_products, enriched_ids = load_checkpoint(output_file)
    pending = [p for p in all_products if p.get("id") not in enriched_ids]

    logger.info(
        f"📊 Total: {len(all_products)} | "
        f"Déjà enrichis: {len(enriched_ids)} | "
        f"À traiter: {len(pending)}"
    )

    if not pending:
        logger.success("🎉 Tous les produits sont déjà enrichis ! Rien à faire.")
        return

    # ── Estimation du temps ──────────────────────────────────────────────────
    secs_per_product = (WAIT_JS + 1_000 + PAGE_TIMEOUT * 0.1) / 1_000 / concurrency
    estimated_minutes = (len(pending) * secs_per_product) / 60
    logger.info(f"⏱️  Temps estimé : ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f}h)")

    # ── Lancement du scraping ────────────────────────────────────────────────
    start_time = datetime.now()
    processed_count = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        semaphore = asyncio.Semaphore(concurrency)

        # Traitement par batchs pour le checkpointing
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(pending) + batch_size - 1) // batch_size

            logger.info(
                f"\n── Batch {batch_num}/{total_batches} "
                f"(produits {batch_start+1}–{batch_start+len(batch)}) ──"
            )

            # Traitement concurrent du batch
            tasks = [process_product(browser, prod, semaphore) for prod in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filtrer les exceptions éventuelles
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Exception non gérée dans gather : {r}")
                else:
                    enriched_products.append(r)

            processed_count += len(batch)

            # ── Checkpoint ────────────────────────────────────────────────────
            save_checkpoint(enriched_products, output_file)
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = processed_count / elapsed * 60 if elapsed > 0 else 0
            remaining = len(pending) - processed_count
            eta_min = (remaining / (speed / 60)) / 60 if speed > 0 else 0

            logger.success(
                f"💾 Checkpoint sauvegardé → {len(enriched_products)} produits total | "
                f"Vitesse : {speed:.0f} prod/min | "
                f"ETA : ~{eta_min:.0f} min restantes"
            )

            # ── Pause anti-ban entre les batchs ───────────────────────────────
            if batch_start + batch_size < len(pending):
                pause = random.uniform(3, 8)
                logger.debug(f"😴 Pause anti-ban : {pause:.1f}s…")
                await asyncio.sleep(pause)

        await browser.close()

    # ── Rapport final ────────────────────────────────────────────────────────
    elapsed_total = (datetime.now() - start_time).total_seconds()
    success_count = sum(
        1 for p in enriched_products
        if p.get("_enrichment_status") == "success"
    )
    with_rating = sum(1 for p in enriched_products if p.get("rating") is not None)

    logger.success("=" * 60)
    logger.success("✅  SCRAPING TERMINÉ !")
    logger.success(f"   Produits traités   : {processed_count}")
    logger.success(f"   Succès             : {success_count}")
    logger.success(f"   Avec rating réel   : {with_rating} ({with_rating/max(len(enriched_products),1)*100:.1f}%)")
    logger.success(f"   Durée totale       : {elapsed_total/60:.1f} minutes")
    logger.success(f"   Fichier final      : {output_file}")
    logger.success("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌙 Overnight Product Enricher")
    parser.add_argument("--input",       default=DEFAULT_INPUT,  help="Fichier JSON d'entrée")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT, help="Fichier JSON de sortie (checkpoint)")
    parser.add_argument("--batch",       type=int, default=BATCH_SAVE_SIZE, help="Taille du batch de sauvegarde")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,     help="Navigateurs parallèles")
    args = parser.parse_args()

    asyncio.run(main(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch,
        concurrency=args.concurrency,
    ))
