"""
Smart eCommerce Intelligence — Kubeflow Pipeline
FST Tanger — LSI2 — DM & SID 2025/2026

Pipeline de production corrigé et optimisé.
Emplacement : mlops/kubeflow/pipeline.py
"""

from collections.abc import Sequence

import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


# ==========================================
# COMPOSANTS DU PIPELINE
# ==========================================

@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3"]
)
def data_cleaning_component(
    input_data: Input[Dataset],
    cleaned_data: Output[Dataset],
    metrics: Output[Metrics]
):
    """Étape 1: Nettoyage des données"""
    import pandas as pd
    import json

    # Charger les données
    with open(input_data.path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    df = pd.DataFrame(products)

    # Nettoyage adaptatif selon les noms de colonnes (id ou product_id)
    initial_count = len(df)
    id_col = 'product_id' if 'product_id' in df.columns else 'id'
    name_col = 'name' if 'name' in df.columns else 'title'

    df = df.drop_duplicates(subset=[id_col])
    df = df.dropna(subset=[name_col, 'price'])

    # Convertir les types
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
    df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce').fillna(0)

    # Filtrer les valeurs aberrantes
    df = df[df['price'] > 0]
    df = df[df['price'] < 10000]
    df = df[df['rating'] >= 0]
    df = df[df['rating'] <= 5]

    final_count = len(df)

    # Sauvegarder
    df.to_csv(cleaned_data.path, index=False)

    # Métriques
    metrics.log_metric("initial_products", initial_count)
    metrics.log_metric("cleaned_products", final_count)
    metrics.log_metric("duplicates_removed", initial_count - final_count)

    print(f"✅ Data cleaning: {initial_count} → {final_count} products")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3", "scikit-learn==1.3.0"]
)
def feature_engineering_component(
    cleaned_data: Input[Dataset],
    features_data: Output[Dataset],
    metrics: Output[Metrics]
):
    """Étape 2: Feature Engineering"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(cleaned_data.path)

    # Calculer le score de popularité
    df['popularity_score'] = (
        df['rating'] * 0.4 +
        np.log1p(df['review_count']) * 0.3 +
        (1 / (df['price'] + 1)) * 100 * 0.3
    )

    # Features pour le ML
    df['price_rating_ratio'] = df['price'] / (df['rating'] + 0.1)
    df['review_density'] = df['review_count'] / (df['price'] + 1)
    df['is_high_rated'] = (df['rating'] >= 4.5).astype(int)
    df['is_popular'] = (df['review_count'] >= df['review_count'].median()).astype(int)
    df['is_success'] = (df['is_popular'] & df['is_high_rated']).astype(int)

    # Normalisation (CORRIGÉ : Syntaxe propre sans crash de dictionnaire implicite)
    scaler = StandardScaler()
    numeric_cols = ['price', 'rating', 'review_count', 'popularity_score']
    scaled_features = scaler.fit_transform(df[numeric_cols])
    for i, col in enumerate(numeric_cols):
        df[f'{col}_normalized'] = scaled_features[:, i]

    # Sauvegarder
    df.to_csv(features_data.path, index=False)

    # Métriques
    metrics.log_metric("features_created", len(df.columns))
    metrics.log_metric("avg_popularity_score", float(df['popularity_score'].mean()))

    print(f"✅ Feature engineering: {len(df.columns)} features created")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3", "scikit-learn==1.3.0"]
)
def clustering_component(
    features_data: Input[Dataset],
    clustered_data: Output[Dataset],
    metrics: Output[Metrics]
):
    """Étape 3: Clustering (KMeans + DBSCAN + Isolation Forest)"""
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    df = pd.read_csv(features_data.path)

    # Features pour clustering
    feature_cols = ['price', 'rating', 'review_count', 'popularity_score']
    X = df[feature_cols].fillna(0)

    # KMeans
    n_clusters = min(4, len(df) // 5) if len(df) > 5 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans.fit_predict(X)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    df['cluster_dbscan'] = dbscan.fit_predict(X)

    # Isolation Forest (détection d'anomalies)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['is_anomaly'] = iso_forest.fit_predict(X)
    df['anomaly_score'] = iso_forest.score_samples(X)

    # PCA pour visualization
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]

    # Sauvegarder
    df.to_csv(clustered_data.path, index=False)

    # Métriques de qualité si taille suffisante
    if len(df) > n_clusters and df['cluster_kmeans'].nunique() > 1:
        silhouette_kmeans = silhouette_score(X, df['cluster_kmeans'])
        davies_bouldin_kmeans = davies_bouldin_score(X, df['cluster_kmeans'])
        metrics.log_metric("silhouette_score_kmeans", float(silhouette_kmeans))
        metrics.log_metric("davies_bouldin_kmeans", float(davies_bouldin_kmeans))

    metrics.log_metric("n_clusters_kmeans", n_clusters)
    metrics.log_metric("anomalies_detected", int((df['is_anomaly'] == -1).sum()))
    metrics.log_metric("pca_variance_explained", float(sum(pca.explained_variance_ratio_)))

    print(f"✅ Clustering terminé.")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "numpy==1.24.3"]
)
def topk_selection_component(
    clustered_data: Input[Dataset],
    topk_data: Output[Dataset],
    topk: int = 30
):
    """Étape 4: Sélection des Top-K produits"""
    import pandas as pd

    df = pd.read_csv(clustered_data.path)
    top_products = df.nlargest(min(topk, len(df)), 'popularity_score')
    top_products.to_csv(topk_data.path, index=False)
    print(f"✅ Top-K selection effectuée.")


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.0.3", "mlxtend==0.23.1"]
)
def association_rules_component(
    clustered_data: Input[Dataset],
    rules_data: Output[Dataset],
    metrics: Output[Metrics]
):
    """Étape 5: Règles d'association (Apriori)"""
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules

    df = pd.read_csv(clustered_data.path)

    # Fallback propre si pas de colonnes de shop
    cat_col = 'category' if 'category' in df.columns else None
    shop_col = 'shop_name' if 'shop_name' in df.columns else ('source_platform' if 'source_platform' in df.columns else None)

    if cat_col and shop_col:
        basket = df.groupby([shop_col, cat_col]).size().unstack(fill_value=0)
        basket = (basket > 0).astype(int)

        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False).head(100)
            rules.to_csv(rules_data.path, index=False)
            metrics.log_metric("rules_found", len(rules))
            return

    pd.DataFrame().to_csv(rules_data.path, index=False)
    metrics.log_metric("rules_found", 0)


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.0.3", "numpy==1.24.3", "scikit-learn==1.3.0",
        "xgboost==2.0.3", "joblib==1.3.2"
    ]
)
def model_training_component(
    features_data: Input[Dataset],
    rf_model: Output[Model],
    xgb_model: Output[Model],
    metrics: Output[Metrics]
):
    """Étape 6: Entraînement et évaluation des modèles"""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
    import joblib
    import os

    # ----- Fonctions utilitaires internes (Kubeflow scope) -----
    def _safe_stratify(y):
        return y if y.nunique() > 1 and int(y.value_counts().min()) >= 2 else None

    def _optimal_f1_threshold(y_true, y_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        if thresholds.size == 0:
            return 0.5, 0.0

        precision = precision[:-1]
        recall = recall[:-1]
        denominator = precision + recall
        f1_scores = np.divide(2 * precision * recall, denominator,
                              out=np.zeros_like(denominator),
                              where=denominator > 0)
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds[best_idx]), float(f1_scores[best_idx])

    def _predict_with_threshold(y_proba, threshold):
        return (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)

    # ------------------------------------------------------------

    df = pd.read_csv(features_data.path)

    if len(df) < 10:
        print("⚠️ Données insuffisantes pour l'entraînement.")
        pd.DataFrame().to_csv(rf_model.path)
        pd.DataFrame().to_csv(xgb_model.path)
        return

    target_column = next((col for col in ['is_success', 'is_top_product'] if col in df.columns), None)
    if target_column is None:
        raise ValueError("Aucune cible supervisée trouvée. Le composant refuse d'inventer is_success à partir des features.")

    leakage_columns = {
        'price', 'rating', 'review_count', 'popularity_score', 'price_rating_ratio',
        'review_density', 'is_high_rated', 'is_popular', 'price_normalized',
        'rating_normalized', 'review_count_normalized', 'popularity_score_normalized',
        target_column,
    }
    feature_cols = [c for c in df.columns if c not in leakage_columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("Aucune feature exploitable disponible après exclusion des colonnes à risque.")

    X = df[feature_cols].fillna(0)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=_safe_stratify(y_train),
    )

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_fit, y_fit)
    threshold_rf, _ = _optimal_f1_threshold(y_val, rf.predict_proba(X_val)[:, 1])
    rf.fit(X_train, y_train)
    y_pred_rf = _predict_with_threshold(rf.predict_proba(X_test)[:, 1], threshold_rf)

    os.makedirs(os.path.dirname(rf_model.path), exist_ok=True)
    joblib.dump(rf, rf_model.path)

    # XGBoost
    xgb_mod = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_mod.fit(X_fit, y_fit)
    threshold_xgb, _ = _optimal_f1_threshold(y_val, xgb_mod.predict_proba(X_val)[:, 1])
    xgb_mod.fit(X_train, y_train)
    y_pred_xgb = _predict_with_threshold(xgb_mod.predict_proba(X_test)[:, 1], threshold_xgb)

    os.makedirs(os.path.dirname(xgb_model.path), exist_ok=True)
    joblib.dump(xgb_mod, xgb_model.path)

    metrics.log_metric("rf_f1_score", float(f1_score(y_test, y_pred_rf, zero_division=0)))
    metrics.log_metric("xgb_f1_score", float(f1_score(y_test, y_pred_xgb, zero_division=0)))
    print("✅ Entraînement des modèles terminé avec succès.")


# ==========================================
# DÉFINITION DU PIPELINE
# ==========================================

@dsl.pipeline(
    name="smart-ecommerce-intelligence-pipeline",
    description="Complete ML pipeline for product analysis, clustering, and Top-K selection"
)
def smart_ecommerce_pipeline(
    input_data_path: str = "/app/data/raw/products_enriched_overnight.json",
    topk: int = 30
):
    cleaning_task = data_cleaning_component(input_data=input_data_path)

    features_task = feature_engineering_component(
        cleaned_data=cleaning_task.outputs['cleaned_data']
    )

    clustering_task = clustering_component(
        features_data=features_task.outputs['features_data']
    )

    topk_task = topk_selection_component(
        clustered_data=clustering_task.outputs['clustered_data'],
        topk=topk
    )

    rules_task = association_rules_component(
        clustered_data=clustering_task.outputs['clustered_data']
    )

    training_task = model_training_component(
        features_data=features_task.outputs['features_data']
    )


# ==========================================
# COMPILATION
# ==========================================

if __name__ == "__main__":
    from kfp import compiler
    import os

    os.makedirs('mlops/kubeflow', exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path='mlops/kubeflow/pipeline.yaml'
    )
    print("✅ Pipeline compilé avec succès dans : mlops/kubeflow/pipeline.yaml")