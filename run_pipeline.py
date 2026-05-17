#!/usr/bin/env python
"""
run_pipeline.py
================
Point d'entrée unique et robuste du projet.
Enchaîne le scraping asynchrone (Module Agents) et le pipeline complet
de Data Mining (Module ML) en alignant le fichier de sortie pour le serveur MCP.

FIX v2 : ne renomme plus products_final.csv → le dashboard a besoin de ce fichier.
         Crée simplement products_processed.csv comme copie/lien si nécessaire.
"""

import asyncio
import sys
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import orchestrate, save_to_json, _build_sources_from_env

try:
    import run_local
    RUN_LOCAL_AVAILABLE = True
except ImportError:
    RUN_LOCAL_AVAILABLE = False


async def main():
    load_dotenv()
    start_time = time.time()

    logger.info("=== [STAGE 1] Collecte & Enrichissement Asynchrone (Agents A2A) ===")
    sources = _build_sources_from_env()
    if not sources:
        logger.warning("Aucune source configurée dans le .env. Annulation du run.")
        return

    products = await orchestrate(sources)
    if not products:
        logger.error("Le scraping n'a extrait aucun produit. Fin du pipeline.")
        return

    json_path = save_to_json(products)
    logger.success(f"Fichier brut généré avec succès : {json_path}")

    logger.info("=== [STAGE 2] Exécution du Pipeline de Data Mining & ML ===")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # FIX : products_final.csv est le fichier canonique du dashboard.
    # run_local.py le génère. On crée products_processed.csv comme copie
    # pour la rétrocompatibilité MCP, SANS supprimer products_final.csv.
    final_csv   = processed_dir / "products_final.csv"
    processed_csv = processed_dir / "products_processed.csv"

    if RUN_LOCAL_AVAILABLE:
        try:
            run_local.start_ml_pipeline(input_file=str(json_path), top_k=20)
        except Exception as e:
            logger.error(f"Le traitement ML interne a échoué : {e}")
            return
    else:
        logger.info("Exécution via le module de clustering direct...")
        try:
            import numpy as np
            import pandas as pd
            from ml.clustering import cluster_products

            df_raw = pd.read_json(json_path)
            artifact_dir = PROJECT_ROOT / "artifacts"
            artifact_dir.mkdir(exist_ok=True)

            for col in ["price", "review_count", "rating"]:
                if col not in df_raw.columns:
                    df_raw[col] = 0.0

            df_raw["log_price"] = np.log1p(df_raw["price"])
            df_raw["rating_score"] = df_raw["rating"] / 5.0
            df_raw["popularity_score"] = df_raw["review_count"]
            df_raw["discount_score"] = 0.0

            df_final = cluster_products(df_raw, artifact_paths=artifact_dir)
            df_final.to_csv(final_csv, index=False)
            logger.success("Traitement ML direct terminé.")
        except Exception as e:
            logger.exception(f"Le fallback ML direct a échoué : {e}")
            return

    # FIX : copier products_final.csv → products_processed.csv (sans renommer)
    if final_csv.exists():
        shutil.copy2(final_csv, processed_csv)
        logger.success(f"✅ products_final.csv conservé pour le dashboard")
        logger.success(f"✅ products_processed.csv mis à jour pour le serveur MCP")
    else:
        logger.warning("products_final.csv non trouvé après le pipeline ML.")

    logger.success(f"✨ Pipeline global exécuté en {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main())