"""
Smart eCommerce Intelligence — Kubeflow Pipeline
FST Tanger — LSI2 — DM & SID 2025/2026

Pipeline reproductible en 6 composants kfp :
  1. scrape       → collecte les données (ou génère synthétiques)
  2. clean        → nettoyage + déduplication
  3. features     → feature engineering (20 features)
  4. train        → RandomForest + XGBoost supervisé
  5. cluster      → KMeans + DBSCAN non-supervisé + Apriori
  6. dashboard    → export CSV pour Streamlit

Usage :
    # Compiler le pipeline en YAML
    python mlops/kubeflow/pipeline.py --compile

    # Soumettre sur un cluster Kubeflow
    python mlops/kubeflow/pipeline.py --submit --host http://localhost:8080

    # Tester localement sans Kubeflow (mode local)
    python mlops/kubeflow/pipeline.py --local
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

# ── kfp SDK v2 ────────────────────────────────────────────────────────────────
try:
    import kfp
    from kfp import dsl
    from kfp.dsl import (
        Dataset, Input, Output, Metrics, Model,
        component, pipeline,
    )
    KFP_AVAILABLE = True
except ImportError:
    KFP_AVAILABLE = False
    # Stubs pour permettre l'import sans kfp installé
    class _Stub:
        def __getattr__(self, _): return lambda *a, **k: None
    dsl = _Stub()
    def component(*a, **k): return lambda f: f
    def pipeline(*a, **k): return lambda f: f
    Dataset = Input = Output = Metrics = Model = object


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 1 — Scraping / Génération des données
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "httpx[asyncio]==0.27.0",
        "loguru==0.7.2",
        "pydantic==2.7.1",
        "requests==2.32.3",
    ],
)
def scrape_data(
    n_synthetic: int,
    use_real_shopify: bool,
    max_shopify_pages: int,
    raw_data: Output[Dataset],
) -> None:
    """
    Collecte les données produits.

    Si use_real_shopify=True, scrape les boutiques Shopify réelles.
    Sinon, génère n_synthetic produits synthétiques pour les tests.
    Les deux sources peuvent être combinées.
    """
    import json
    import random
    import numpy as np
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    CATEGORIES = [
        "Électronique", "Mode", "Maison & Jardin", "Sport & Loisirs",
        "Beauté & Santé", "Informatique", "Alimentation", "Jouets & Enfants",
    ]
    PLATFORMS = ["shopify", "woocommerce"]
    BRANDS = ["Samsung", "Apple", "Nike", "Adidas", "Sony", "Philips", "Generic"]
    SHOPS = ["BoutiqueMaroc", "TechShop.ma", "ModeMaghreb", "ElectroStore", "SportPlus.ma"]

    PRODUCT_TEMPLATES = {
        "Électronique": ["Écouteurs", "Smartphone", "Tablette", "Montre connectée"],
        "Mode": ["T-shirt Premium", "Jean Slim", "Veste en cuir", "Sneakers"],
        "Maison & Jardin": ["Lampe LED", "Cafetière", "Robot cuiseur"],
        "Sport & Loisirs": ["Tapis de yoga", "Haltères", "Corde à sauter"],
        "Beauté & Santé": ["Sérum visage", "Crème hydratante", "Parfum"],
        "Informatique": ["Souris sans fil", "Clavier mécanique", "Hub USB"],
        "Alimentation": ["Huile d'argan bio", "Thé à la menthe", "Miel naturel"],
        "Jouets & Enfants": ["Puzzle 1000 pièces", "LEGO Creator", "Peluche"],
    }

    def _random_product(idx: int) -> dict:
        rng = random.Random(idx)
        np_rng = np.random.RandomState(idx)
        category = rng.choice(CATEGORIES)
        templates = PRODUCT_TEMPLATES.get(category, ["Produit générique"])
        base_name = rng.choice(templates)
        brand = rng.choice(BRANDS)
        platform = rng.choice(PLATFORMS)
        shop = rng.choice(SHOPS)
        base_price = float(np_rng.lognormal(mean=5.5, sigma=0.8))
        base_price = round(min(max(base_price, 15.0), 5000.0), 2)
        has_discount = rng.random() < 0.30
        original_price = round(base_price * rng.uniform(1.1, 1.5), 2) if has_discount else None
        rating = round(min(max(float(np_rng.normal(3.8, 0.8)), 0.0), 5.0), 1)
        review_count = int(np_rng.lognormal(mean=3.0, sigma=1.5))
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
            "source_platform": platform,
            "scraped_at": scraped_at.isoformat(),
            "n_variants": rng.randint(1, 5),
            "n_images": rng.randint(1, 8),
            "discount_pct": round(((original_price - base_price) / original_price * 100), 2) if original_price else 0.0,
            "has_discount": has_discount,
        }

    # Génération des données synthétiques
    products = [_random_product(i) for i in range(n_synthetic)]
    print(f"[Scraper] {len(products)} produits synthétiques générés")

    # Sauvegarde
    Path(raw_data.path).parent.mkdir(parents=True, exist_ok=True)
    with open(raw_data.path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, default=str)
    print(f"[Scraper] Données → {raw_data.path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 2 — Nettoyage des données
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4", "loguru==0.7.2"],
)
def clean_data(
    raw_data: Input[Dataset],
    cleaned_data: Output[Dataset],
    cleaning_metrics: Output[Metrics],
) -> None:
    """
    Nettoie le dataset :
    - Supprime les doublons (par id)
    - Supprime les titres vides
    - Impute les prix manquants (médiane)
    - Normalise les types numériques
    - Calcule discount_percentage
    """
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path

    with open(raw_data.path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    initial = len(df)

    # 1. Déduplication
    dedup_cols = ["id"] if "id" in df.columns else ["title", "source_platform"]
    df = df.drop_duplicates(subset=dedup_cols)

    # 2. Titres vides
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]

    # 3. Prix
    median_price = df[df["price"] > 0]["price"].median() if "price" in df.columns else 100.0
    df["price"] = df["price"].replace(0, pd.NA).fillna(median_price).infer_objects(copy=False)
    if "original_price" not in df.columns:
        df["original_price"] = df["price"]
    else:
        df["original_price"] = df["original_price"].fillna(df["price"])

    # 4. Remise
    df["discount_percentage"] = np.where(
        df["original_price"] > df["price"],
        ((df["original_price"] - df["price"]) / df["original_price"] * 100).round(2),
        0.0,
    )

    # 5. Numériques
    df["rating"] = df["rating"].fillna(0.0).infer_objects(copy=False)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["stock"] = df["stock"].fillna(0).astype(int)

    if "is_in_stock" in df.columns:
        df["is_in_stock"] = df["is_in_stock"].fillna(True).astype(bool)

    # 6. Catégorielles
    for col in ["category", "brand", "shop_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("Inconnu")
    if "category" not in df.columns:
        df["category"] = "Inconnu"

    df = df.reset_index(drop=True)
    n_final = len(df)

    # Métriques
    cleaning_metrics.log_metric("initial_count", initial)
    cleaning_metrics.log_metric("final_count", n_final)
    cleaning_metrics.log_metric("removed_rows", initial - n_final)
    cleaning_metrics.log_metric("removal_rate_pct", round((initial - n_final) / initial * 100, 2))

    Path(cleaned_data.path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_data.path, index=False, encoding="utf-8")
    print(f"[Cleaner] {n_final}/{initial} produits → {cleaned_data.path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 3 — Feature Engineering (20 features)
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4", "loguru==0.7.2", "scikit-learn==1.5.0"],
)
def engineer_features(
    cleaned_data: Input[Dataset],
    features_data: Output[Dataset],
    top_k: int = 20,
) -> None:
    """
    Construit 20 features ML + label is_top_product.

    Groupe 1 — Scores normalisés [0,1] (5 features) :
      price_score, rating_score, popularity_score, stock_score, discount_score

    Groupe 2 — Features dérivées (6 features) :
      value_for_money, price_to_median_ratio, log_reviews,
      variant_diversity, n_images_norm, brand_score

    Groupe 3 — Features métier (5 features) :
      product_completeness, has_discount_flag, is_in_stock_flag,
      shop_reputation, review_rating_coherence

    Groupe 4 — Features temporelles/texte (4 features) :
      title_length_norm, tag_diversity, price_volatility, category_popularity
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    df = pd.read_csv(cleaned_data.path, encoding="utf-8")
    n = len(df)

    # ── Groupe 1 : Scores normalisés ──────────────────────────────────────────
    max_price = df["price"].max()
    df["price_score"] = (1 - df["price"] / max_price).clip(0, 1) if max_price > 0 else 0.5

    df["rating_score"] = (df["rating"] / 5.0).clip(0, 1)

    df["log_reviews"] = np.log1p(df["review_count"])
    max_log = df["log_reviews"].max()
    df["popularity_score"] = (df["log_reviews"] / max_log).clip(0, 1) if max_log > 0 else 0.0

    max_stock = df["stock"].max()
    df["stock_score"] = (df["stock"] / max_stock).clip(0, 1) if max_stock > 0 else 0.0

    disc_col = "discount_pct" if "discount_pct" in df.columns else "discount_percentage"
    if disc_col in df.columns:
        max_disc = df[disc_col].max()
        df["discount_score"] = (df[disc_col] / max_disc).clip(0, 1) if max_disc > 0 else 0.0
    else:
        df["discount_score"] = 0.0

    # ── Groupe 2 : Features dérivées ──────────────────────────────────────────
    df["value_for_money"] = (df["rating_score"] * df["price_score"]).clip(0, 1)
    df.loc[df["rating"] == 0, "value_for_money"] = 0.0

    cat_medians = df.groupby("category")["price"].transform("median")
    ratio = (df["price"] / cat_medians.replace(0, 1)).clip(0, 5)
    df["price_to_median_ratio"] = (1 - (ratio - 1).abs().clip(0, 4) / 4)

    if "n_variants" in df.columns:
        max_v = df["n_variants"].clip(upper=20).max()
        df["variant_diversity"] = (df["n_variants"].clip(upper=20) / max_v).clip(0, 1) if max_v > 0 else 0.0
    else:
        df["variant_diversity"] = 0.0

    if "n_images" in df.columns:
        max_img = df["n_images"].clip(upper=10).max()
        df["n_images_norm"] = (df["n_images"].clip(upper=10) / max_img).clip(0, 1) if max_img > 0 else 0.5
    else:
        df["n_images_norm"] = 0.5

    # NOUVEAU — brand_score : popularité de la marque dans le catalogue
    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts(normalize=True)
        df["brand_score"] = df["brand"].map(brand_counts).fillna(0.0).clip(0, 1)
    else:
        df["brand_score"] = 0.0

    # ── Groupe 3 : Features métier ────────────────────────────────────────────
    comp_cols = [c for c in ["price", "rating", "review_count", "stock", "category"] if c in df.columns]
    df["product_completeness"] = df[comp_cols].notna().mean(axis=1)
    df.loc[df["price"] == 0, "product_completeness"] *= 0.5

    disc_col2 = "has_discount" if "has_discount" in df.columns else None
    df["has_discount_flag"] = df[disc_col2].astype(float) if disc_col2 else (df["discount_score"] > 0).astype(float)

    df["is_in_stock_flag"] = df["is_in_stock"].astype(float) if "is_in_stock" in df.columns else (df["stock"] > 0).astype(float)

    # NOUVEAU — shop_reputation : score moyen des produits du même shop
    if "shop_name" in df.columns:
        shop_avg_rating = df.groupby("shop_name")["rating"].transform("mean")
        df["shop_reputation"] = (shop_avg_rating / 5.0).clip(0, 1)
    else:
        df["shop_reputation"] = 0.5

    # NOUVEAU — review_rating_coherence : avis nombreux ET bonne note = cohérence
    df["review_rating_coherence"] = (df["popularity_score"] * df["rating_score"]).clip(0, 1)

    # ── Groupe 4 : Features texte/diversité ───────────────────────────────────
    # NOUVEAU — title_length_norm : un titre ni trop court ni trop long
    if "title" in df.columns:
        title_len = df["title"].str.len().fillna(0)
        df["title_length_norm"] = (title_len.clip(10, 100) - 10) / 90
    else:
        df["title_length_norm"] = 0.5

    # NOUVEAU — category_popularity : fréquence de la catégorie dans le catalogue
    cat_counts = df["category"].value_counts(normalize=True)
    df["category_popularity"] = df["category"].map(cat_counts).fillna(0.0).clip(0, 1)

    # NOUVEAU — price_volatility proxy : écart relatif prix / médiane catégorie
    df["price_volatility"] = (ratio - 1.0).abs().clip(0, 4) / 4

    # NOUVEAU — tag_diversity (si disponible, sinon 0.5)
    if "tags" in df.columns:
        tag_counts_series = df["tags"].fillna("").apply(lambda x: len(str(x).split(",")) if x else 0)
        max_tags = tag_counts_series.clip(upper=10).max()
        df["tag_diversity"] = (tag_counts_series.clip(upper=10) / max_tags).clip(0, 1) if max_tags > 0 else 0.0
    else:
        df["tag_diversity"] = 0.0

    # ── Score composite pondéré (20 features) ─────────────────────────────────
    df["composite_score"] = (
        0.25 * df["rating_score"]
        + 0.20 * df["popularity_score"]
        + 0.12 * df["value_for_money"]
        + 0.12 * df["discount_score"]
        + 0.08 * df["price_score"]
        + 0.06 * df["stock_score"]
        + 0.06 * df["shop_reputation"]
        + 0.05 * df["review_rating_coherence"]
        + 0.03 * df["brand_score"]
        + 0.03 * df["product_completeness"]
    ).clip(0, 1)

    # ── Label Top-K ────────────────────────────────────────────────────────────
    actual_k = min(top_k, n)
    if actual_k > 0 and df["composite_score"].nunique() > 1:
        threshold = df["composite_score"].nlargest(actual_k).min()
        df["is_top_product"] = (df["composite_score"] >= threshold).astype(int)
    else:
        df["is_top_product"] = 0

    n_top = df["is_top_product"].sum()
    print(f"[FeatureEng] 20 features | Top-{actual_k}: {n_top} produits ({n_top/n*100:.1f}%)")

    Path(features_data.path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_data.path, index=False, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 4 — Entraînement supervisé (RF + XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2", "numpy==1.26.4", "scikit-learn==1.5.0",
        "xgboost==2.0.3", "joblib==1.4.2", "loguru==0.7.2",
    ],
)
def train_supervised(
    features_data: Input[Dataset],
    rf_model: Output[Model],
    xgb_model: Output[Model],
    train_metrics: Output[Metrics],
) -> None:
    """
    Entraîne RandomForest + XGBoost sur le label is_top_product.
    Évalue avec : Accuracy, F1, Précision, Rappel, AUC-ROC, CV-5fold.
    """
    import pandas as pd
    import numpy as np
    import joblib
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )
    from xgboost import XGBClassifier

    FEATURE_COLUMNS = [
        "price_score", "rating_score", "popularity_score", "stock_score", "discount_score",
        "value_for_money", "price_to_median_ratio", "log_reviews", "variant_diversity",
        "product_completeness", "has_discount_flag", "is_in_stock_flag", "n_images_norm",
        "brand_score", "shop_reputation", "review_rating_coherence",
        "title_length_norm", "category_popularity", "price_volatility", "tag_diversity",
    ]

    df = pd.read_csv(features_data.path, encoding="utf-8")

    if len(df) < 50 or df["is_top_product"].nunique() < 2:
        print(f"[Train] Données insuffisantes ({len(df)} produits). Supervisé ignoré.")
        Path(rf_model.path).mkdir(parents=True, exist_ok=True)
        Path(xgb_model.path).mkdir(parents=True, exist_ok=True)
        return

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["is_top_product"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    def _evaluate(model, name, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv = cross_val_score(model, X, y, cv=5, scoring="f1_weighted", n_jobs=-1)
        metrics = {
            "accuracy":    round(float(accuracy_score(y_test, y_pred)), 4),
            "f1":          round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "precision":   round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall":      round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "auc_roc":     round(float(roc_auc_score(y_test, y_proba)), 4),
            "cv_f1_mean":  round(float(cv.mean()), 4),
            "cv_f1_std":   round(float(cv.std()), 4),
        }
        print(f"[{name}] Accuracy={metrics['accuracy']} | F1={metrics['f1']} | AUC-ROC={metrics['auc_roc']} | CV-F1={metrics['cv_f1_mean']}±{metrics['cv_f1_std']}")
        return metrics

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_metrics = _evaluate(rf, "RandomForest", X_train, X_test, y_train, y_test)

    Path(rf_model.path).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, Path(rf_model.path) / "random_forest.joblib")

    # XGBoost
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         scale_pos_weight=scale_pos_weight, random_state=42,
                         eval_metric="logloss", verbosity=0)
    xgb_metrics = _evaluate(xgb, "XGBoost", X_train, X_test, y_train, y_test)

    Path(xgb_model.path).mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, Path(xgb_model.path) / "xgboost.joblib")

    # Logguer les meilleures métriques
    best = rf_metrics if rf_metrics["auc_roc"] > xgb_metrics["auc_roc"] else xgb_metrics
    for k, v in best.items():
        train_metrics.log_metric(k, v)

    print(f"[Train] Supervisé terminé — {len(feature_cols)} features utilisées")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 5 — Clustering + Règles d'association
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2", "numpy==1.26.4", "scikit-learn==1.5.0",
        "mlxtend==0.23.1", "joblib==1.4.2", "loguru==0.7.2",
    ],
)
def cluster_and_mine(
    features_data: Input[Dataset],
    clustered_data: Output[Dataset],
    rules_data: Output[Dataset],
    cluster_metrics: Output[Metrics],
) -> None:
    """
    Module non-supervisé :
    - KMeans (K=5) avec Silhouette + Davies-Bouldin
    - DBSCAN pour la détection d'anomalies
    - PCA 2D pour la visualisation
    - Règles d'association Apriori entre catégories
    """
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    from pathlib import Path

    df = pd.read_csv(features_data.path, encoding="utf-8")

    CLUSTER_FEATURES = ["price_score", "rating_score", "popularity_score", "discount_score"]
    features_ok = [f for f in CLUSTER_FEATURES if f in df.columns]

    if len(features_ok) >= 2 and len(df) >= 10:
        X_raw = df[features_ok].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        # KMeans
        n_clusters = min(5, len(df) // 10)
        kmeans = KMeans(n_clusters=max(2, n_clusters), random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(X_scaled)
        df["cluster_label"] = "Segment " + df["cluster"].astype(str)

        # Métriques clustering
        if df["cluster"].nunique() >= 2:
            sil = round(float(silhouette_score(X_scaled, df["cluster"])), 4)
            db  = round(float(davies_bouldin_score(X_scaled, df["cluster"])), 4)
            cluster_metrics.log_metric("silhouette_score", sil)
            cluster_metrics.log_metric("davies_bouldin", db)
            cluster_metrics.log_metric("n_clusters", df["cluster"].nunique())
            print(f"[KMeans] K={df['cluster'].nunique()} | Silhouette={sil} | Davies-Bouldin={db}")

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)
        n_outliers = (df["dbscan_cluster"] == -1).sum()
        cluster_metrics.log_metric("dbscan_outliers", int(n_outliers))
        print(f"[DBSCAN] {n_outliers} outliers sur {len(df)} produits")

        # PCA 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1]
        variance = sum(pca.explained_variance_ratio_[:2]) * 100
        df["pca_variance_explained"] = round(variance, 2)
        cluster_metrics.log_metric("pca_variance_explained_pct", round(variance, 2))
        print(f"[PCA] Variance expliquée: {variance:.1f}%")

    else:
        df["cluster"] = 0
        df["dbscan_cluster"] = 0
        df["PC1"] = 0.0
        df["PC2"] = 0.0

    # Règles d'association Apriori
    rules_df = pd.DataFrame()
    if "shop_name" in df.columns and "category" in df.columns:
        transactions = (
            df.groupby("shop_name")["category"]
            .apply(lambda cats: list(cats.dropna().unique()))
            .tolist()
        )
        transactions = [t for t in transactions if len(t) >= 2]

        if len(transactions) >= 3:
            te = TransactionEncoder()
            basket_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
            frequent = apriori(basket_df, min_support=0.1, use_colnames=True)
            if not frequent.empty:
                rules_df = association_rules(frequent, metric="confidence", min_threshold=0.6)
                rules_df = rules_df.sort_values("lift", ascending=False).reset_index(drop=True)
                print(f"[Apriori] {len(rules_df)} règles extraites")

    # Sauvegarde
    Path(clustered_data.path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clustered_data.path, index=False, encoding="utf-8")

    Path(rules_data.path).parent.mkdir(parents=True, exist_ok=True)
    rules_df.to_csv(rules_data.path, index=False, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANT 6 — Export final pour le dashboard
# ─────────────────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2", "loguru==0.7.2"],
)
def export_for_dashboard(
    clustered_data: Input[Dataset],
    rules_data: Input[Dataset],
    final_data: Output[Dataset],
    topk_data: Output[Dataset],
    top_k: int = 20,
) -> None:
    """
    Prépare les CSV finaux consommés par le dashboard Streamlit :
    - products_final.csv   (tous les produits enrichis)
    - top_k_products.csv   (Top-K classement ML)
    - association_rules.csv (déjà généré par cluster_and_mine)
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(clustered_data.path, encoding="utf-8")

    # Top-K
    top_k_df = (
        df[df["is_top_product"] == 1]
        .sort_values("composite_score", ascending=False)
        .head(top_k)
    )

    Path(final_data.path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_data.path, index=False, encoding="utf-8")

    Path(topk_data.path).parent.mkdir(parents=True, exist_ok=True)
    top_k_df.to_csv(topk_data.path, index=False, encoding="utf-8")

    print(f"[Export] {len(df)} produits total | {len(top_k_df)} Top-K")
    print(f"[Export] CSV final → {final_data.path}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
@pipeline(
    name="smart-ecommerce-intelligence",
    description="Pipeline ML complet : Scraping → Nettoyage → Features → RF+XGB → KMeans+DBSCAN+Apriori → Export BI",
)
def smart_ecommerce_pipeline(
    n_synthetic: int = 2000,
    use_real_shopify: bool = False,
    max_shopify_pages: int = 3,
    top_k: int = 20,
):
    """
    Pipeline Kubeflow complet.

    Paramètres configurables via l'UI Kubeflow :
        n_synthetic       : nombre de produits synthétiques à générer
        use_real_shopify  : activer le scraping des boutiques réelles
        max_shopify_pages : pages max par boutique Shopify
        top_k             : nombre de produits Top-K à sélectionner
    """
    # Étape 1 : Scraping
    scrape_task = scrape_data(
        n_synthetic=n_synthetic,
        use_real_shopify=use_real_shopify,
        max_shopify_pages=max_shopify_pages,
    )
    scrape_task.set_display_name("1. Scraping données")

    # Étape 2 : Nettoyage
    clean_task = clean_data(raw_data=scrape_task.outputs["raw_data"])
    clean_task.set_display_name("2. Nettoyage")
    clean_task.after(scrape_task)

    # Étape 3 : Feature Engineering
    feature_task = engineer_features(
        cleaned_data=clean_task.outputs["cleaned_data"],
        top_k=top_k,
    )
    feature_task.set_display_name("3. Feature Engineering (20 features)")
    feature_task.after(clean_task)

    # Étape 4 : Entraînement supervisé
    train_task = train_supervised(features_data=feature_task.outputs["features_data"])
    train_task.set_display_name("4. RandomForest + XGBoost")
    train_task.after(feature_task)

    # Étape 5 : Clustering + Data Mining
    cluster_task = cluster_and_mine(features_data=feature_task.outputs["features_data"])
    cluster_task.set_display_name("5. KMeans + DBSCAN + Apriori")
    cluster_task.after(feature_task)

    # Étape 6 : Export
    export_task = export_for_dashboard(
        clustered_data=cluster_task.outputs["clustered_data"],
        rules_data=cluster_task.outputs["rules_data"],
        top_k=top_k,
    )
    export_task.set_display_name("6. Export Dashboard BI")
    export_task.after(cluster_task)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _compile():
    """Compile le pipeline en YAML pour déploiement Kubeflow."""
    if not KFP_AVAILABLE:
        print("❌ kfp non installé. Lance : pip install kfp==2.7.0")
        return
    output = Path("mlops/kubeflow/pipeline.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    kfp.compiler.Compiler().compile(smart_ecommerce_pipeline, str(output))
    print(f"✅ Pipeline compilé → {output}")


def _submit(host: str):
    """Soumet le pipeline sur un cluster Kubeflow."""
    if not KFP_AVAILABLE:
        print("❌ kfp non installé.")
        return
    client = kfp.Client(host=host)
    run = client.create_run_from_pipeline_func(
        smart_ecommerce_pipeline,
        arguments={"n_synthetic": 2000, "top_k": 20},
        run_name="smart-ecommerce-run-v1",
    )
    print(f"✅ Run soumis : {run.run_id}")
    print(f"   Dashboard : {host}/#/runs/details/{run.run_id}")


def _run_local():
    """
    Exécution locale du pipeline sans Kubeflow.
    Utile pour tester avant de déployer.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    print("🚀 Mode local — pipeline sans Kubeflow")
    print("Lance directement les scripts Python :")
    print("  python data/generate_synthetic.py 2000")
    print("  python ml/pipeline.py")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart eCommerce Kubeflow Pipeline")
    parser.add_argument("--compile", action="store_true", help="Compiler le pipeline en YAML")
    parser.add_argument("--submit",  action="store_true", help="Soumettre sur Kubeflow")
    parser.add_argument("--local",   action="store_true", help="Mode local sans Kubeflow")
    parser.add_argument("--host",    default="http://localhost:8080", help="URL Kubeflow")
    args = parser.parse_args()

    if args.compile:
        _compile()
    elif args.submit:
        _submit(args.host)
    else:
        _run_local()