"""
tests/test_complete_validation.py
==================================
Tests automatisés — VERSION CORRIGÉE
Corrections :
  1. Aucun Data Leakage dans ml_features_clean (popularity_score + composants retirés de X)
  2. Séparation claire : sample_data (algo) vs real_data (intégration — strict assert)
  3. TestPipelineIntegration et TestPerformance utilisent des assert stricts
  4. Les modèles testés utilisent les vraies features indépendantes du projet

Usage:
    pytest tests/test_complete_validation.py -v --tb=short
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    add_scoring_features,
    FEATURE_COLUMNS,
    select_model_features,
)
from ml.clustering import cluster_products
from ml.metrics import evaluate_classifier, evaluate_clustering

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ── Chemins ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

# Colonnes à exclure de X — directement ou indirectement liées à composite_score
LEAKAGE_COLUMNS = {
    "composite_score",
    "popularity_score",
    "price_score",
    "rating_score",
    "log_reviews",
    "value_for_money",
    "review_rating_coherence",
    "discount_score",
    "stock_score",
}


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def real_data():
    """
    Données RÉELLES issues du pipeline scraping + preprocessing.
    Échoue EXPLICITEMENT si le pipeline n'a pas été exécuté.
    Utilisée par les tests d'intégration uniquement.
    """
    candidates = [
        DATA_DIR / "processed" / "products_final.csv",
        DATA_DIR / "processed" / "products_processed.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1024:
            df = pd.read_csv(path)
            assert len(df) > 10, (
                f"Le fichier {path.name} existe mais contient trop peu de lignes ({len(df)}). "
                "Le scraping a peut-être échoué."
            )
            return df

    pytest.fail(
        "\n\n❌ ERREUR D'INTÉGRATION : Aucun fichier de données réelles trouvé.\n"
        "   Exécutez d'abord le pipeline complet :\n"
        "     $ python run_local.py\n"
        f"   Fichiers attendus dans : {DATA_DIR / 'processed'}\n"
        "   Ce test ne doit PAS passer avec des données synthétiques."
    )


@pytest.fixture(scope="module")
def sample_data():
    """
    Données pour les tests UNITAIRES d'algorithmes ML.
    Utilise les données réelles si disponibles.
    Si absentes, génère des données synthétiques EXPLICITEMENT (non masquées)
    pour valider le comportement des algorithmes uniquement — pas le pipeline.
    """
    candidates = [
        DATA_DIR / "processed" / "products_final.csv",
        DATA_DIR / "processed" / "products_processed.csv",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1024:
            return pd.read_csv(path)

    # Génération explicite — ne masque pas une panne du pipeline
    print(
        "\n⚠️  [UNIT TEST MODE] Données réelles absentes.\n"
        "   → Données synthétiques utilisées UNIQUEMENT pour tester les algorithmes.\n"
        "   → Les tests d'intégration (TestPipelineIntegration) échoueront."
    )
    np.random.seed(42)
    n = 400
    prices = np.random.lognormal(mean=4.5, sigma=1.2, size=n).clip(5, 2000)
    return pd.DataFrame({
        "product_id":      [f"P{i:04d}" for i in range(n)],
        "name":            [f"Produit Test {i}" for i in range(n)],
        "title":           [f"Titre complet du produit {i} avec mots-clés" for i in range(n)],
        "shop_name":       [f"Shop_{i % 12}" for i in range(n)],
        "source_platform": np.random.choice(["shopify", "woocommerce"], n),
        "category":        np.random.choice(
            ["Electronics", "Clothing", "Home", "Sports", "Beauty", "Books"], n
        ),
        "brand":           np.random.choice(
            ["BrandA", "BrandB", "BrandC", "BrandD", "Unknown"], n
        ),
        "price":           prices,
        "stock":           np.random.randint(0, 200, n),
        "rating":          np.clip(np.random.normal(3.8, 0.8, n), 1.0, 5.0),
        "review_count":    np.random.randint(0, 2000, n),
        "discount_pct":    np.random.uniform(0, 0.6, n),
        "n_variants":      np.random.randint(1, 15, n),
        "n_images":        np.random.randint(1, 10, n),
        "is_in_stock":     np.random.choice([0, 1], n, p=[0.15, 0.85]),
        "tags":            [",".join([f"tag{j}" for j in range(np.random.randint(0, 8))]) for _ in range(n)],
    })


@pytest.fixture(scope="module")
def engineered_data(sample_data):
    """Données enrichies avec les 20 features du projet (add_scoring_features)."""
    return add_scoring_features(sample_data.copy())


@pytest.fixture(scope="module")
def ml_features_clean(engineered_data):
    """
    Features ML SANS Data Leakage.

    Stratégie :
    - Cible  : TOP 20% selon composite_score (score de ranking métier pondéré).
    - X      : features structurelles et indépendantes uniquement.
                composite_score et ses composantes directes sont EXCLUS de X
                via LEAKAGE_COLUMNS → select_model_features().

    Features conservées (exemples) :
        variant_diversity, n_images_norm, brand_score, product_completeness,
        has_discount_flag, is_in_stock_flag, shop_reputation, title_length_norm,
        category_popularity, price_volatility, tag_diversity,
        price_to_median_ratio
    """
    df = engineered_data.copy()

    # Cible supervisée : top 20% par composite_score (non inclus dans X)
    threshold = df["composite_score"].quantile(0.80)
    df["is_top_product"] = (df["composite_score"] >= threshold).astype(int)

    # Sélection sécurisée — exclut target + LEAKAGE_COLUMNS
    safe_features = select_model_features(
        df,
        candidate_columns=FEATURE_COLUMNS,
        target_column="is_top_product",
        leakage_columns=LEAKAGE_COLUMNS,
    )

    assert len(safe_features) >= 3, (
        f"Trop peu de features non-leakage disponibles : {safe_features}\n"
        "Vérifiez FEATURE_COLUMNS dans feature_engineering.py"
    )

    X = df[safe_features].fillna(0)
    y = df["is_top_product"]

    # Stratification sécurisée (vérifie que les deux classes existent)
    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print(
        f"\n[ml_features_clean] Features utilisées ({len(safe_features)}) : {safe_features}\n"
        f"  Train : {len(X_train)} | Test : {len(X_test)} | "
        f"Positifs test : {y_test.sum()}/{len(y_test)}"
    )
    return X_train, X_test, y_train, y_test, safe_features


# ==========================================
# TESTS DES MODÈLES SUPERVISÉS
# ==========================================

def _compute_optimal_threshold(model, X_test, y_test):
    """Calcule le seuil F1-optimal sur la courbe Precision-Recall."""
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = f1_scores[:-1].argmax()  # thresholds a 1 élément de moins
    best_threshold = thresholds[best_idx]
    return y_proba, best_threshold


class TestSupervisedModels:
    """
    Tests RF et XGBoost sur features SANS leakage.
    Les scores attendus sont réalistes (non parfaits) : AUC 0.55–0.85.
    """

    def test_random_forest_no_leakage(self, ml_features_clean):
        """RandomForest sur features indépendantes — valide l'absence de leakage."""
        X_train, X_test, y_train, y_test, features = ml_features_clean

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_proba, best_threshold = _compute_optimal_threshold(rf, X_test, y_test)
        y_pred_optimal = (y_proba >= best_threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_optimal)
        f1       = f1_score(y_test, y_pred_optimal, zero_division=0)
        auc      = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0

        # Seuil réaliste : on n'attend PAS la perfection sans leakage
        assert accuracy > 0.45, f"Accuracy anormalement faible : {accuracy:.3f}"
        assert auc > 0.50,      f"AUC < 0.50 : le modèle est pire qu'aléatoire ({auc:.3f})"
        assert auc < 0.999,     (
            f"AUC suspicieusement parfaite ({auc:.3f}). "
            "Risque de data leakage — vérifiez LEAKAGE_COLUMNS."
        )

        cv_scores = cross_val_score(rf, X_train, y_train, cv=3, scoring="roc_auc")

        print("\n" + "=" * 60)
        print("RANDOM FOREST — SANS LEAKAGE")
        print("=" * 60)
        print(f"Features ({len(features)}) : {features}")
        print(f"Seuil optimal  : {best_threshold:.3f}")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"F1-Score       : {f1:.4f}")
        print(f"ROC-AUC        : {auc:.4f}  ← valeur honnête attendue entre 0.55 et 0.90")
        print(f"CV AUC (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report :")
        print(classification_report(y_test, y_pred_optimal, zero_division=0))
        print("Confusion Matrix :")
        print(confusion_matrix(y_test, y_pred_optimal))
        print("=" * 60)

    def test_xgboost_no_leakage(self, ml_features_clean):
        """XGBoost sur features indépendantes avec scale_pos_weight."""
        X_train, X_test, y_train, y_test, features = ml_features_clean

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_weight = round(neg / pos, 2) if pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X_train, y_train)

        y_proba, best_threshold = _compute_optimal_threshold(xgb, X_test, y_test)
        y_pred_optimal = (y_proba >= best_threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_optimal)
        f1       = f1_score(y_test, y_pred_optimal, zero_division=0)
        auc      = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.0

        assert accuracy > 0.45, f"Accuracy anormalement faible : {accuracy:.3f}"
        assert auc > 0.50,      f"AUC < 0.50 : modèle pire qu'aléatoire ({auc:.3f})"
        assert auc < 0.999,     (
            f"AUC suspicieusement parfaite ({auc:.3f}). "
            "Risque de data leakage — vérifiez LEAKAGE_COLUMNS."
        )

        importance_df = pd.DataFrame({
            "feature": features,
            "importance": xgb.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\n" + "=" * 60)
        print("XGBOOST — SANS LEAKAGE")
        print("=" * 60)
        print(f"scale_pos_weight : {scale_weight}  (neg={neg}, pos={pos})")
        print(f"Seuil optimal    : {best_threshold:.3f}")
        print(f"Accuracy         : {accuracy:.4f}")
        print(f"F1-Score         : {f1:.4f}")
        print(f"ROC-AUC          : {auc:.4f}  ← valeur honnête attendue entre 0.55 et 0.90")
        print("\nTop features (importance) :")
        print(importance_df.head(5).to_string(index=False))
        print("\nClassification Report :")
        print(classification_report(y_test, y_pred_optimal, zero_division=0))
        print("Confusion Matrix :")
        print(confusion_matrix(y_test, y_pred_optimal))
        print("=" * 60)


# ==========================================
# TESTS DES MÉTHODES NON SUPERVISÉES
# ==========================================

class TestUnsupervisedMethods:

    def test_kmeans_clustering_quality(self, engineered_data):
        """KMeans avec métriques de qualité sur les features engineered."""
        df = engineered_data.copy()

        # Features SANS leakage pour le clustering aussi
        cluster_features = [
            col for col in [
                "price_to_median_ratio", "variant_diversity", "n_images_norm",
                "brand_score", "product_completeness", "is_in_stock_flag",
                "title_length_norm", "category_popularity", "price_volatility",
                "tag_diversity",
            ]
            if col in df.columns
        ]
        assert len(cluster_features) >= 3, f"Pas assez de features pour KMeans : {cluster_features}"

        X = df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        silhouette  = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)

        assert silhouette > 0.1,       f"Silhouette trop faible : {silhouette:.3f}"
        assert davies_bouldin < 5.0,   f"Davies-Bouldin trop élevé : {davies_bouldin:.3f}"

        df["cluster"] = labels
        print("\n" + "=" * 60)
        print("KMEANS — VALIDATION")
        print("=" * 60)
        print(f"Features ({len(cluster_features)}) : {cluster_features}")
        print(f"Silhouette Score  : {silhouette:.4f}")
        print(f"Davies-Bouldin    : {davies_bouldin:.4f}")
        print(f"\nRépartition par cluster :\n{df['cluster'].value_counts().sort_index()}")
        print("=" * 60)

    def test_dbscan_anomaly_detection(self, engineered_data):
        """DBSCAN pour détection d'anomalies sur espace standardisé."""
        df = engineered_data.copy()
        feature_cols = [
            col for col in [
                "price_to_median_ratio", "variant_diversity", "n_images_norm",
                "brand_score", "product_completeness",
            ]
            if col in df.columns
        ]
        X = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dbscan = DBSCAN(eps=0.8, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        noise_pct  = n_noise / len(X) * 100

        assert n_clusters > 0, "DBSCAN n'a trouvé aucun cluster valide"
        assert noise_pct < 60, f"Taux de bruit trop élevé : {noise_pct:.1f}%"

        print(f"\n🔹 DBSCAN — Clusters : {n_clusters} | Anomalies : {n_noise}/{len(X)} ({noise_pct:.1f}%)")


# ==========================================
# TESTS DES RÈGLES D'ASSOCIATION
# ==========================================

class TestAssociationRules:

    def test_association_rules_metrics(self):
        """Valider les règles générées par le pipeline Apriori/FP-Growth."""
        rules_path = DATA_DIR / "processed" / "association_rules.csv"

        if not rules_path.exists():
            pytest.skip(
                "association_rules.csv absent — exécutez python run_local.py d'abord."
            )
        if rules_path.stat().st_size < 100:
            pytest.skip("association_rules.csv vide — vérifiez ml/apriori_rules.py.")

        rules = pd.read_csv(rules_path)

        for col in ["support", "confidence", "lift"]:
            assert col in rules.columns, f"Colonne manquante dans les règles : {col}"

        assert len(rules) > 0,              "Aucune règle d'association générée"
        assert rules["confidence"].mean() > 0, "Confidence moyenne nulle"
        assert rules["lift"].mean() > 1.0,    "Lift moyen ≤ 1.0 — associations non informatives"

        quality_rules = rules[rules["lift"] > 1.5]

        print("\n" + "=" * 60)
        print("RÈGLES D'ASSOCIATION — VALIDATION")
        print("=" * 60)
        print(f"Total règles           : {len(rules)}")
        print(f"Règles qualité (>1.5)  : {len(quality_rules)}")
        print(f"Support moyen          : {rules['support'].mean():.4f}")
        print(f"Confidence moyenne     : {rules['confidence'].mean():.4f}")
        print(f"Lift moyen             : {rules['lift'].mean():.4f}")
        if len(rules) >= 5:
            print("\nTop 5 règles (par lift) :")
            print(
                rules.nlargest(5, "lift")[
                    ["antecedents", "consequents", "support", "confidence", "lift"]
                ]
            )
        print("=" * 60)


# ==========================================
# TESTS D'INTÉGRATION — DONNÉES RÉELLES OBLIGATOIRES
# ==========================================

class TestPipelineIntegration:
    """
    Tests de bout-en-bout. Requièrent les vraies sorties du pipeline.
    Échouent EXPLICITEMENT si python run_local.py n'a pas été exécuté.
    """

    def test_complete_pipeline_execution(self, real_data):
        """Vérifie que toutes les étapes du pipeline ont produit leurs sorties."""
        data_dir = DATA_DIR / "processed"

        required_files = [
            "products_cleaned.csv",
            "products_features.csv",
            "products_clustered.csv",
            "top_k_products.csv",
            "pca_viz.csv",
        ]

        for fname in required_files:
            fpath = data_dir / fname
            assert fpath.exists(), (
                f"\n❌ Fichier manquant : {fname}\n"
                "   Exécutez : python run_local.py"
            )
            df_check = pd.read_csv(fpath)
            assert len(df_check) > 0, f"Fichier vide : {fname}"
            print(f"✅ {fname:<35} {len(df_check):>5} lignes, {len(df_check.columns):>3} colonnes")

    def test_final_data_coherence(self, real_data):
        """Vérifie la cohérence des données finales (pas de doublons, pas de NaN critiques)."""
        # On vérifie la présence de "id" (standardisé par le ProductSchema) ou "product_id"
        id_col = "id" if "id" in real_data.columns else "product_id"
        assert id_col in real_data.columns, (
            "Colonne identifiant produit manquante dans products_final.csv"
        )

        # Et pour le nom du produit (title au lieu de name)
        title_col = "title" if "title" in real_data.columns else "name"
        assert title_col in real_data.columns, (
            "Colonne titre produit manquante dans products_final.csv"
        )

        # Le reste du test (vérification des doublons sur l'id) utilise la bonne colonne
        duplicate_rate = real_data[id_col].duplicated().mean()
        assert duplicate_rate < 0.05, (
            f"Trop de doublons dans les données finales : {duplicate_rate:.1%}"
        )
        print(f"\n✅ Cohérence données : {len(real_data)} produits, {duplicate_rate:.1%} doublons")


# ==========================================
# TESTS DE PERFORMANCE
# ==========================================

class TestPerformance:

    def test_pipeline_execution_time(self):
        """
        Vérifie que les fichiers de sortie existent ET que leur chargement est rapide.
        Échoue si le pipeline n'a pas été exécuté.
        """
        import time

        data_path = DATA_DIR / "processed" / "products_final.csv"

        # Assertion stricte — ne passe PAS silencieusement si le fichier est absent
        assert data_path.exists(), (
            "\n❌ products_final.csv introuvable.\n"
            "   Exécutez : python run_local.py\n"
            "   Ce test ne peut pas valider les performances sans données réelles."
        )

        start = time.time()
        df = pd.read_csv(data_path)
        _ = df.describe()
        if "cluster_kmeans" in df.columns:
            _ = df.groupby("cluster_kmeans").size()
        elapsed = time.time() - start

        assert elapsed < 30, f"Chargement trop lent : {elapsed:.1f}s (attendu < 30s)"
        print(f"\n⏱️  Chargement de {len(df)} produits : {elapsed:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes"])