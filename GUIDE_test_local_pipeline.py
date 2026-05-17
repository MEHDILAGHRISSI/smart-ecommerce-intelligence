#!/usr/bin/env python
"""
GUIDE : Script de Test Local du Pipeline Complet
================================================

Ce script enchaîne les étapes du pipeline Smart eCommerce Intelligence :

1️⃣ Scraping asynchrone (orchestrateur) via Shopify/WooCommerce
   - Lance les agents scrapers en parallèle
   - Enrichit les descriptions avec le LLM
   - Accumule les produits enrichis
   - Sauvegarde en JSON brut dans data/raw/

2️⃣ Chargement et préparation des données
   - Lit le JSON généré
   - Sérialise en DataFrame Pandas
   - Affiche un résumé des colonnes disponibles

3️⃣ Entraînement des modèles ML (mode "train")
   - Normalise les features avec StandardScaler
   - Entraîne KMeans pour la segmentation
   - Entraîne DBSCAN pour la détection d'outliers
   - Entraîne IsolationForest pour les anomalies financières
   - Applique PCA pour la réduction de dimension
   - Sauvegarde tous les modèles en .joblib dans artifacts/

4️⃣ Vérification des fichiers persistants
   - Vérifie la présence de scaler.joblib
   - Vérifie la présence de kmeans.joblib
   - Vérifie la présence de dbscan.joblib
   - Vérifie la présence de isolation_forest.joblib
   - Vérifie la présence de pca.joblib
   - Vérifie la présence de metadata.json

5️⃣ Résumé final
   - Distribution des clusters créés
   - Taux d'anomalies détectées
   - Statistiques des scores d'anomalies
   - Confirmation du succès ✅

═════════════════════════════════════════════════════════════════════

📋 UTILISATION
==============

Lancer le script :
  $ python test_local_pipeline.py

Le script va :
  ✅ Créer 561+ produits de test (ou selon l'API)
  ✅ Sauvegarder dans : data/raw/products_*.json
  ✅ Créer des modèles dans : artifacts/
  ✅ Afficher un rapport détaillé
  ✅ Confirmer que tout fonctionne ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ÉTAPES EN DÉTAIL
===================

ÉTAPE 1 : SCRAPING
─────────────────
Le script lance l'orchestrateur asynchrone qui :
  • Initialise plusieurs agents Shopify/WooCommerce en parallèle
  • Scrape les URLs configurées dans les variables d'environnement
  • Enrichit les descriptions via le module LLM
  • Accumule les produits bruts dans une file d'attente asyncio.Queue
  • Sauvegarde le JSON brut une fois la collecte terminée

Entrée  : URLs Shopify/WooCommerce de .env
Sortie  : data/raw/products_*.json (1.6MB ~ 561 produits)

ÉTAPE 2 : CHARGEMENT ET PRÉPARATION
────────────────────────────────────
Le script charge le JSON via Pandas et affiche :
  • Nombre de lignes et colonnes
  • Noms des colonnes (id, title, description, price, etc.)

Entrée  : data/raw/products_*.json
Sortie  : DataFrame Pandas (561 × 24)

ÉTAPE 3 : ENTRAÎNEMENT ML
──────────────────────────
Le script appelle cluster_products() en mode "train" :

  1. StandardScaler normalise les features
     → sauvegarde dans artifacts/scaler.joblib

  2. KMeans segmente les produits (K=3)
     → sauvegarde dans artifacts/kmeans.joblib

  3. DBSCAN détecte les outliers structurels
     → sauvegarde dans artifacts/dbscan.joblib

  4. IsolationForest détecte les anomalies financières
     → sauvegarde dans artifacts/isolation_forest.joblib

  5. PCA réduit à 2 dimensions pour la visualisation
     → sauvegarde dans artifacts/pca.joblib

  6. metadata.json stocke les min/max pour l'inférence
     → sauvegarde dans artifacts/metadata.json

Entrée  : DataFrame brut (561 produits)
Sortie  : DataFrame enrichi + fichiers .joblib

ÉTAPE 4 : VÉRIFICATION DES ARTEFACTS
─────────────────────────────────────
Le script énumère les fichiers créés et affiche :
  ✅ scaler.joblib              (X.X KB)
  ✅ kmeans.joblib              (X.X KB)
  ✅ dbscan.joblib              (X.X KB)
  ✅ isolation_forest.joblib     (X.X KB)
  ✅ pca.joblib                 (X.X KB)
  ✅ metadata.json              (X.X KB)

Si un fichier manque :
  ⚠️ AVERTISSEMENT : Certains artefacts manquent

ÉTAPE 5 : RÉSUMÉ
────────────────
Affichage des statistiques finales :
  • Distribution des clusters (Cluster 0: 187, Cluster 1: 210, ...)
  • Taux d'anomalies global (8.9% du catalogue)
  • Score anomalie moyen et maximum
  • Liste des colonnes ML créées

Sortie : Rapport structuré et formaté

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 EXEMPLE DE SORTIE
====================

$ python test_local_pipeline.py

✅ Variables d'environnement chargées
📍 4 source(s) configurée(s)
🔄 Lancement du scraping asynchrone...
✅ 561 produit(s) collecté(s)
💾 Données brutes sauvegardées → data/raw/products_20260516_121504.json

✅ 561 produit(s) chargé(s)
📊 DataFrame : 561 lignes × 24 colonnes

🚀 Lancement du clustering...
✅ Clustering terminé : 561 produit(s)

📁 Vérification dans : artifacts/
   ✅ scaler.joblib              (15.2 KB)
   ✅ kmeans.joblib              (8.4 KB)
   ✅ dbscan.joblib              (1.2 KB)
   ✅ isolation_forest.joblib     (45.3 KB)
   ✅ pca.joblib                 (3.8 KB)
   ✅ metadata.json              (0.5 KB)

📊 DataSet final : 561 produit(s)

🎯 Distribution des clusters :
   Cluster 0: 187 produit(s) (33.3%)
   Cluster 1: 210 produit(s) (37.4%)
   Cluster 2: 164 produit(s) (29.2%)

🚨 Anomalies détectées : 50 (8.9%)
📈 Score anomalie : moy=0.234, max=0.872

✅ PIPELINE TEST COMPLETED SUCCESSFULLY !

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️  VARIABLES D'ENVIRONNEMENT
=============================

Le script lit depuis .env :

SHOPIFY_ALTERNATIVES = url1,url2,url3
SHOPIFY_API_KEY = votre_clé_shopify

WOO_ALTERNATIVES = url1,url2
WOO_CONSUMER_KEY = votre_clé_woo
WOO_CONSUMER_SECRET = votre_secret_woo

LLM_API_KEY = votre_clé_openai (optionnel)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 FICHIERS GÉNÉRÉS
===================

Après l'exécution, le projet contient :

data/raw/
  └─ products_20260516_121504.json      (1.6 MB, 561 produits)

artifacts/
  ├─ scaler.joblib                      (15.2 KB)
  ├─ kmeans.joblib                      (8.4 KB)
  ├─ dbscan.joblib                      (1.2 KB)
  ├─ isolation_forest.joblib            (45.3 KB)
  ├─ pca.joblib                         (3.8 KB)
  └─ metadata.json                      (0.5 KB)

Ces fichiers peuvent être utilisés en mode "inférence" :

from ml.clustering import cluster_products

df_new = pd.read_csv("new_products.csv")
result = cluster_products(
    df_new,
    scaler_path="artifacts/scaler.joblib",
    models_dir="artifacts/"
)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 CE QUE CELA VALIDE
====================

✅ L'orchestrateur scrape bien en asynchrone
✅ Les données brutes sont correctement sérializées en JSON
✅ Le pipeline ML intègre tous les modèles (KMeans + DBSCAN + IsoForest)
✅ La persistance des artefacts fonctionne (.joblib)
✅ L'inférence peut utiliser ces modèles sauvegardés
✅ Les anomalies sont correctement calculées
✅ Le PCA génère les coordonnées pour la visualisation

Prochaines étapes :
  1. Exécuter la partie dashboard avec `streamlit run dashboard/app.py`
  2. Test MCP avec `python -m mcp_server.server --transport stdio`
  3. Vérification de production du pipeline complet

═════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    import os
    # Afficher ce fichier de documentation
    this_file = os.path.abspath(__file__)
    with open(this_file, "r", encoding="utf-8") as f:
        print(f.read())

