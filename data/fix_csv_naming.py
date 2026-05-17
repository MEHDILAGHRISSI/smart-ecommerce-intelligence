"""
data/fix_csv_naming.py
======================
Script one-shot pour corriger le problème de nommage CSV.

Problème : run_pipeline.py renomme products_final.csv → products_processed.csv
           mais le dashboard cherche products_final.csv.

Solution permanente : le dashboard app.py (v2) a déjà le fallback intégré.
Ce script crée un symlink ou copie pour assurer la compatibilité.

Usage :
    python data/fix_csv_naming.py
"""

import shutil
from pathlib import Path

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def fix():
    final = DATA_PROCESSED / "products_final.csv"
    processed = DATA_PROCESSED / "products_processed.csv"

    if final.exists():
        print(f"✅ products_final.csv existe déjà ({final.stat().st_size // 1024} KB) — rien à faire.")
        return

    if processed.exists():
        shutil.copy2(processed, final)
        print(f"✅ Copie créée : products_processed.csv → products_final.csv ({final.stat().st_size // 1024} KB)")
    else:
        print("⚠️  Ni products_final.csv ni products_processed.csv n'existe.")
        print("    Lance d'abord : python run_local.py")


if __name__ == "__main__":
    fix()