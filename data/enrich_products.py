import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright
from loguru import logger


async def get_product_details(browser, product):
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    page = await context.new_page()
    url = product["product_url"]

    try:
        # NOUVEAU CODE : On attend que la page soit complètement chargée
        await page.goto(url, wait_until="load", timeout=60000)

        # Astuce cruciale : On force une pause de 3 secondes (3000 millisecondes)
        # pour laisser le JavaScript des applications tierces injecter les notes et le JSON-LD
        await page.wait_for_timeout(3000)

        # (Optionnel) Scroller un peu vers le bas, car certaines apps d'avis
        # ne se chargent que quand l'utilisateur commence à scroller (Lazy Loading)
        await page.mouse.wheel(0, 500)
        await page.wait_for_timeout(1000)


        # 1. NOUVELLE MÉTHODE : Extraction via les données structurées SEO (JSON-LD)
        # C'est la méthode la plus professionnelle pour récupérer les vraies notes
        rating_data = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script[type="application/ld+json"]');
            for (let script of scripts) {
                try {
                    const data = JSON.parse(script.innerText);
                    // Le JSON-LD peut être un objet direct ou un tableau
                    const items = Array.isArray(data) ? data : [data];
                    for (let item of items) {
                        // On cherche l'entité Produit qui contient les avis
                        if (item['@type'] === 'Product' && item.aggregateRating) {
                            return {
                                rating: parseFloat(item.aggregateRating.ratingValue),
                                count: parseInt(item.aggregateRating.reviewCount)
                            };
                        }
                    }
                } catch(e) {}
            }
            return null;
        }""")

        if rating_data:
            rating = rating_data['rating']
            review_count = rating_data['count']
        else:
            # Fallback sur les sélecteurs visuels classiques si le JSON-LD n'existe pas
            rating = await page.evaluate("""() => {
                const el = document.querySelector('.jdgm-prev-badge__avg-rating, .spr-badge-caption, .okeReviews-starRating--star');
                return el ? parseFloat(el.innerText.match(/(\d+\.?\d*)/)[0]) : null;
            }""")
            review_count = None

        # 2. Extraction du Stock (inchangé)
        is_instock = await page.evaluate("""() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            const addBtn = buttons.find(b => b.innerText.toLowerCase().includes('add') || b.innerText.toLowerCase().includes('ajouter'));
            return addBtn ? !addBtn.disabled : true;
        }""")

        # 2. L'ULTIME HACK DU STOCK EXACT (The Cart Hack via AJAX)
        exact_stock = await page.evaluate("""async () => {
                    try {
                        // 1. Trouver l'ID de la variante actuellement sélectionnée sur la page
                        const idInput = document.querySelector('form[action^="/cart/add"] [name="id"], select[name="id"]');
                        if (!idInput) return null;
                        const variantId = idInput.value;

                        // 2. Tenter d'ajouter 9999 unités au panier via l'API AJAX native de Shopify
                        const res = await fetch('/cart/add.js', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ items: [{ id: variantId, quantity: 9999 }] })
                        });
                        const data = await res.json();

                        // 3. Analyser la réponse du serveur Shopify
                        // Erreur 422 = Refus d'ajout (dépassement de stock)
                        if (res.status === 422 && data.description) {
                            // Le message contient le stock exact: "You can only add 14 of this item..."
                            const match = data.description.match(/\d+/);
                            return match ? parseInt(match[0]) : 0;
                        }

                        // Si la requête réussit (status 200), c'est soit un produit digital, 
                        // soit la boutique autorise les précommandes ("Continue selling when out of stock")
                        if (res.ok) {
                            // On nettoie notre trace en vidant le panier
                            await fetch('/cart/clear.js', { method: 'POST' });
                            return 9999; // 9999 signifie "Stock illimité ou Précommande"
                        }
                    } catch (e) {
                        return null;
                    }
                    return null;
                }""")

        product["rating"] = rating
        product["review_count"] = review_count  # On récupère aussi le nombre d'avis !
        product["is_in_stock"] = is_instock

        product["stock"] = 1 if is_instock else 0

        logger.info(f"✅ Enrichi : {product['title']} | Rating: {rating} ({review_count} avis)")

    except Exception as e:
        logger.error(f"⚠️ Erreur légère sur {url} (Ignorée pour continuer)")
    finally:
        await page.close()
        await context.close()

async def main():
    input_file = "data/raw/products_20260330_154810_real_shopify.json"
    with open(input_file, "r") as f:
        products = json.load(f)

    # Pour ne pas être banni, on traite par lots (ex: les 100 premiers pour tester)
    sample = products[:10]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # On lance 5 navigateurs en parallèle pour aller plus vite
        semaphore = asyncio.Semaphore(5)

        async def sem_task(prod):
            async with semaphore:
                await get_product_details(browser, prod)

        await asyncio.gather(*(sem_task(p) for p in sample))
        await browser.close()

    # Sauvegarde
    output_path = "data/raw/products_enriched.json"
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Terminé ! Fichier enrichi dispo dans {output_path}")


if __name__ == "__main__":
    asyncio.run(main())