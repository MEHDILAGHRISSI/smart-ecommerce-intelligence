"""
Générateur de données synthétiques — pour tester le pipeline ML
quand les boutiques de démo retournent peu de produits.

Usage :
    python data/generate_synthetic.py          # 500 produits par défaut
    python data/generate_synthetic.py 2000     # 2000 produits
"""

from __future__ import annotations
import json
import random
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
from loguru import logger

# Import du dossier standardisé
from configs.settings import DATA_RAW_DIR

CATEGORIES = [
    "Électronique", "Mode", "Maison & Jardin", "Sport & Loisirs",
    "Beauté & Santé", "Informatique", "Alimentation", "Jouets & Enfants",
    "Automobile", "Livres & Médias",
]

PLATFORMS = ["shopify", "woocommerce"]

PRODUCT_TEMPLATES = {
    "Électronique": ["Écouteurs", "Smartphone", "Tablette", "Montre connectée",
                     "Chargeur rapide", "Câble USB-C", "Batterie externe"],
    "Mode": ["T-shirt Premium", "Jean Slim", "Veste en cuir", "Sneakers",
             "Robe d'été", "Manteau laine", "Chaussures de ville"],
    "Maison & Jardin": ["Lampe LED", "Coussin déco", "Plante artificielle",
                        "Cafetière", "Robot cuiseur", "Aspirateur"],
    "Sport & Loisirs": ["Tapis de yoga", "Haltères", "Vélo stationnaire",
                        "Corde à sauter", "Gants de boxe", "Protège-tibias"],
    "Beauté & Santé": ["Sérum visage", "Crème hydratante", "Parfum",
                       "Brosse électrique", "Complément alimentaire"],
    "Informatique": ["Souris sans fil", "Clavier mécanique", "Webcam HD",
                     "Hub USB", "SSD externe", "Écran portable"],
    "Alimentation": ["Huile d'argan bio", "Thé à la menthe", "Miel naturel",
                     "Dattes Medjool", "Épices mélangées"],
    "Jouets & Enfants": ["Puzzle 1000 pièces", "LEGO Creator", "Peluche",
                         "Jeu de société", "Voiture télécommandée"],
    "Automobile": ["Organiseur de coffre", "Caméra de recul", "Tapis voiture",
                   "Chargeur allume-cigare", "Désodorisant"],
    "Livres & Médias": ["Roman bestseller", "Livre de cuisine", "BD",
                        "Manga tome 1", "Guide de voyage"],
}

BRANDS = ["Samsung", "Apple", "Nike", "Adidas", "L'Oréal", "Philips",
          "Casio", "Sony", "Nestlé", "Generic", "ProLine", "EcoPlus"]

SHOPS = ["BoutiqueMaroc", "TechShop.ma", "ModeMaghreb", "ElectroStore",
         "MaisonDeco.ma", "SportPlus.ma", "BeautyMaroc", "FoodCorner"]


def _random_product(idx: int) -> dict:
    """Génère un produit synthétique réaliste."""
    rng = random.Random(idx)
    np_rng = np.random.RandomState(idx)

    category = rng.choice(CATEGORIES)
    templates = PRODUCT_TEMPLATES.get(category, ["Produit générique"])
    base_name = rng.choice(templates)
    brand = rng.choice(BRANDS)
    platform = rng.choice(PLATFORMS)
    shop = rng.choice(SHOPS)

    # Prix avec distribution log-normale (réaliste)
    base_price = float(np_rng.lognormal(mean=5.5, sigma=0.8))
    base_price = round(min(max(base_price, 15.0), 5000.0), 2)

    # Remise aléatoire (30% des produits en promo)
    has_discount = rng.random() < 0.30
    original_price = round(base_price * rng.uniform(1.1, 1.5), 2) if has_discount else None

    # Note : centrée sur 3.8, std 0.8
    rating = round(min(max(float(np_rng.normal(3.8, 0.8)), 0.0), 5.0), 1)
    review_count = int(np_rng.lognormal(mean=3.0, sigma=1.5))

    # Stock
    stock = int(np_rng.lognormal(mean=3.5, sigma=1.2)) if rng.random() < 0.85 else 0

    scraped_at = datetime.now(timezone.utc) - timedelta(minutes=rng.randint(0, 60))

    return {
        "id": f"{platform[:3]}-{idx:05d}",
        "title": f"{base_name} {brand} — Ref.{idx:04d}",
        "price": base_price,
        "original_price": original_price,
        "currency": "MAD",
        "rating": rating,
        "review_count": review_count,
        "stock": stock,
        "is_in_stock": stock > 0,
        "category": category,
        "brand": brand,
        "shop_name": shop,
        "description": f"Produit de qualité supérieure dans la catégorie {category}.",
        "image_url": f"https://picsum.photos/seed/{idx}/400/400",
        "product_url": f"https://{shop.lower()}.ma/products/{base_name.lower().replace(' ', '-')}-{idx}",
        "source_platform": platform,
        "scraped_at": scraped_at.isoformat(),
    }


def generate(n: int = 500) -> list[dict]:
    """Génère n produits synthétiques."""
    products = [_random_product(i) for i in range(n)]
    logger.success(f"{n} produits synthétiques générés")
    return products


def save(products: list[dict]) -> Path:
    """Sauvegarde dans le dossier raw configuré avec un timestamp."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Formatage du nom de fichier pour qu'il soit détecté par load_latest_raw_products()
    path = DATA_RAW_DIR / f"raw_products_{ts}_synthetic.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2, default=str)

    logger.success(f"Données sauvegardées → {path}")
    return path


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    logger.info(f"Démarrage de la génération de {n} produits synthétiques...")
    products = generate(n)
    save(products)
    logger.info("Maintenant lance : python ml/pipeline.py")