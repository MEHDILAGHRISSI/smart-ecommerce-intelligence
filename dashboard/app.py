"""
Smart eCommerce Intelligence Dashboard
FST Tanger — LSI2 — DM & SID 2025/2026

Corrections v3 :
  - Import du routeur LLM via generate_response (alias call_llm)
  - Suppression de detect_active_providers → liste statique des providers supportés
  - Adaptation de _run_llm pour utiliser generate_response avec prompt unique (concaténation historique + système)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.settings import DATA_PROCESSED_DIR, Settings
from llm.llm_router import generate_response

# ── Configuration page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Chargement et validation des données
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["price", "title"]


@st.cache_data(ttl=300)
def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Charge les CSV depuis DATA_PROCESSED_DIR.

    FIX : fallback products_processed.csv si products_final.csv absent
    (run_pipeline.py renomme products_final.csv → products_processed.csv).
    """
    # Chercher products_final.csv d'abord, puis fallback
    final_path = DATA_PROCESSED_DIR / "products_final.csv"
    if not final_path.exists():
        fallback = DATA_PROCESSED_DIR / "products_processed.csv"
        if fallback.exists():
            final_path = fallback
        else:
            return {k: None for k in ["final", "topk", "rules", "pca"]}

    paths = {
        "final": final_path,
        "topk":  DATA_PROCESSED_DIR / "top_k_products.csv",
        "rules": DATA_PROCESSED_DIR / "association_rules.csv",
        "pca":   DATA_PROCESSED_DIR / "pca_viz.csv",
    }

    try:
        df_final = pd.read_csv(paths["final"])
    except Exception:
        return {k: None for k in ["final", "topk", "rules", "pca"]}

    missing = [c for c in REQUIRED_COLUMNS if c not in df_final.columns]
    if missing:
        return {k: None for k in ["final", "topk", "rules", "pca"]}

    result: Dict[str, Optional[pd.DataFrame]] = {
        "final": df_final, "topk": None, "rules": None, "pca": None
    }

    try:
        result["topk"] = pd.read_csv(paths["topk"]) if paths["topk"].exists() else df_final.head(20)
    except Exception:
        result["topk"] = df_final.head(20)

    try:
        result["rules"] = pd.read_csv(paths["rules"]) if paths["rules"].exists() else pd.DataFrame()
        if result["rules"] is not None and result["rules"].empty:
            result["rules"] = pd.DataFrame()
    except Exception:
        result["rules"] = pd.DataFrame()

    try:
        result["pca"] = pd.read_csv(paths["pca"]) if paths["pca"].exists() else None
    except Exception:
        result["pca"] = None

    return result


def _safe_series(df: pd.DataFrame, column: str, default: object = 0) -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def _anomaly_detector_series(df: pd.DataFrame) -> pd.Series:
    if "dbscan_outlier" in df.columns:
        db = pd.to_numeric(df["dbscan_outlier"], errors="coerce").fillna(0).astype(int) == 1
    elif "dbscan_cluster" in df.columns:
        db = pd.to_numeric(df["dbscan_cluster"], errors="coerce").fillna(0).astype(int) == -1
    else:
        db = pd.Series(False, index=df.index)

    if "iforest_outlier" in df.columns:
        iforest = pd.to_numeric(df["iforest_outlier"], errors="coerce").fillna(0).astype(int) == 1
    elif "price_anomaly_flag" in df.columns:
        iforest = pd.to_numeric(df["price_anomaly_flag"], errors="coerce").fillna(0).astype(int) == 1
    else:
        iforest = pd.Series(False, index=df.index)

    detector = np.where(
        db & iforest, "DBSCAN + Isolation Forest",
        np.where(db, "DBSCAN", np.where(iforest, "Isolation Forest", "Aucun"))
    )
    return pd.Series(detector, index=df.index)


def _build_pca_scatter(df_pca: pd.DataFrame) -> go.Figure:
    plot_df = df_pca.copy()
    if "cluster_label" not in plot_df.columns and "cluster" in plot_df.columns:
        plot_df["cluster_label"] = "Cluster " + plot_df["cluster"].astype(str)

    plot_df["cluster_label"] = _safe_series(plot_df, "cluster_label", "Inconnu").astype(str).fillna("Inconnu")
    plot_df["title"] = _safe_series(plot_df, "title", "")
    plot_df["price"] = pd.to_numeric(_safe_series(plot_df, "price", 0.0), errors="coerce").fillna(0.0)
    plot_df["review_count"] = pd.to_numeric(_safe_series(plot_df, "review_count", 0.0), errors="coerce").fillna(0.0)
    plot_df["price_anomaly_score"] = pd.to_numeric(_safe_series(plot_df, "price_anomaly_score", 0.0), errors="coerce").fillna(0.0)
    plot_df["is_anomaly"] = pd.to_numeric(_safe_series(plot_df, "is_anomaly", 0), errors="coerce").fillna(0).astype(int)
    plot_df["detector_model"] = _anomaly_detector_series(plot_df)

    fig = go.Figure()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Dark24 + px.colors.qualitative.Pastel

    normals = plot_df[plot_df["is_anomaly"] == 0]
    anomalies = plot_df[plot_df["is_anomaly"] == 1]

    for i, label in enumerate(sorted(normals["cluster_label"].astype(str).unique().tolist())):
        subset = normals[normals["cluster_label"].astype(str) == label]
        customdata = np.stack([
            subset["title"].astype(str),
            subset["cluster_label"].astype(str),
            subset["price"].to_numpy(),
            subset["review_count"].to_numpy(),
            subset["price_anomaly_score"].to_numpy(),
            subset["detector_model"].astype(str).to_numpy(),
        ], axis=-1) if len(subset) else None
        fig.add_trace(go.Scatter(
            x=subset["PC1"], y=subset["PC2"],
            mode="markers", name=label,
            marker=dict(size=9, color=palette[i % len(palette)], opacity=0.7,
                        line=dict(width=0.5, color="white")),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Prix: %{customdata[2]:.2f}<br>"
                "Avis: %{customdata[3]:,.0f}<br>"
                "Score anomalie: %{customdata[4]:.3f}<br>"
                "Détecteur: %{customdata[5]}<extra></extra>"
            ),
        ))

    if not anomalies.empty:
        customdata = np.stack([
            anomalies["title"].astype(str),
            anomalies["cluster_label"].astype(str),
            anomalies["price"].to_numpy(),
            anomalies["review_count"].to_numpy(),
            anomalies["price_anomaly_score"].to_numpy(),
            anomalies["detector_model"].astype(str).to_numpy(),
        ], axis=-1)
        fig.add_trace(go.Scatter(
            x=anomalies["PC1"], y=anomalies["PC2"],
            mode="markers", name="Anomalies",
            marker=dict(size=13, color="#FF4B4B", symbol="x",
                        line=dict(width=1.5, color="#7A0000"), opacity=0.95),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Prix: %{customdata[2]:.2f}<br>"
                "Avis: %{customdata[3]:,.0f}<br>"
                "Score anomalie: %{customdata[4]:.3f}<br>"
                "Détecteur: %{customdata[5]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=560, template="plotly_white",
        legend_title_text="Cluster / Anomalie",
        xaxis_title="Composante PCA 1",
        yaxis_title="Composante PCA 2",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def _anomaly_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    suspect = df.copy()
    if "is_anomaly" in suspect.columns:
        suspect = suspect[pd.to_numeric(suspect["is_anomaly"], errors="coerce").fillna(0).astype(int) == 1]
    elif "dbscan_outlier" in suspect.columns or "iforest_outlier" in suspect.columns:
        db = pd.to_numeric(suspect.get("dbscan_outlier", pd.Series(0, index=suspect.index)), errors="coerce").fillna(0).astype(int) == 1
        ifo = pd.to_numeric(suspect.get("iforest_outlier", pd.Series(0, index=suspect.index)), errors="coerce").fillna(0).astype(int) == 1
        suspect = suspect[db | ifo]
    else:
        return suspect.iloc[0:0]
    if "price_anomaly_score" in suspect.columns:
        suspect = suspect.sort_values("price_anomaly_score", ascending=False)
    return suspect


# ─────────────────────────────────────────────────────────────────────────────
# Chargement initial
# ─────────────────────────────────────────────────────────────────────────────
data = load_data()
df_full = data["final"]

if df_full is None:
    st.sidebar.markdown("## 🛠️ Diagnostic rapide")
    st.sidebar.write(f"Répertoire attendu : `{DATA_PROCESSED_DIR}`")
    st.sidebar.write("Fichiers présents :")
    try:
        existing = [p.name for p in sorted(DATA_PROCESSED_DIR.glob("*"))]
    except Exception:
        existing = []
    for p in existing:
        st.sidebar.write(f"- {p}")
    st.error(
        "⚠️ Données manquantes ou corrompues.\n\n"
        "Génère d'abord les CSV attendus puis relance le dashboard."
    )
    st.code("python data/generate_synthetic.py 500\npython run_local.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — filtres
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=70)
st.sidebar.markdown("## 🛒 Smart eCommerce")
st.sidebar.markdown("*FST Tanger — LSI2 — DM & SID*")
st.sidebar.markdown("---")

all_cats  = sorted(df_full["category"].dropna().unique().tolist()) if "category" in df_full.columns else []
all_shops = sorted(df_full["shop_name"].dropna().unique().tolist()) if "shop_name" in df_full.columns else []
all_plats = df_full["source_platform"].dropna().unique().tolist() if "source_platform" in df_full.columns else []

selected_cats  = st.sidebar.multiselect("Catégories", all_cats, default=[])
selected_shops = st.sidebar.multiselect("Boutiques", all_shops[:20], default=[])
selected_plats = st.sidebar.multiselect("Plateformes", all_plats, default=all_plats)

try:
    price_series = pd.to_numeric(df_full["price"], errors="coerce").dropna()
    if price_series.empty:
        raise ValueError
    min_p, max_p = float(price_series.min()), float(price_series.max())
    if min_p >= max_p:
        min_p, max_p = max(0.0, min_p - 1.0), max_p + 1.0
except Exception:
    min_p, max_p = 0.0, 1000.0
    st.sidebar.warning("Valeurs de prix invalides — slider fixé sur [0,1000].")

price_range = st.sidebar.slider(
    "Fourchette de prix", min_p, max_p, (min_p, max_p),
    step=max(1.0, (max_p - min_p) / 100)
)
stock_only = st.sidebar.checkbox("En stock uniquement", value=False)
st.sidebar.markdown("---")
st.sidebar.caption(f"📦 {len(df_full):,} produits chargés")

# Application des filtres
df = df_full.copy()
if selected_cats and "category" in df.columns:
    df = df[df["category"].isin(selected_cats)]
if selected_shops and "shop_name" in df.columns:
    df = df[df["shop_name"].isin(selected_shops)]
if selected_plats and "source_platform" in df.columns:
    df = df[df["source_platform"].isin(selected_plats)]

try:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
except Exception:
    st.warning("Impossible d'appliquer le filtre prix.")

if stock_only:
    if "is_in_stock" in df.columns:
        df = df[df["is_in_stock"] == True]
    elif "stock" in df.columns:
        df = df[pd.to_numeric(df["stock"], errors="coerce").fillna(0) > 0]

if df.empty:
    st.warning("⚠️ Aucun produit après filtrage. Ajuste les filtres.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    label="Navigation",
    options=[
        "📊 Vue Globale", "🏆 Top-K Produits", "🏪 Shops & Géo",
        "🔵 Clustering & PCA", "🔗 Règles d'Association", "🤖 Assistant LLM"
    ],
    index=0,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Vue Globale
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Vue Globale":
    st.title("📊 Vue Globale — KPIs")
    st.caption(f"Données filtrées : **{len(df):,} produits** sur {len(df_full):,} total")

    n_topk = int(df["is_top_product"].sum()) if "is_top_product" in df.columns else 0
    n_shops = df["shop_name"].nunique() if "shop_name" in df.columns else df["source_platform"].nunique() if "source_platform" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Produits", f"{len(df):,}")
    c2.metric("💰 Prix Moyen", f"{df['price'].mean():.2f}" if "price" in df.columns else "N/A")
    c3.metric("⭐ Note Moy.", f"{df['rating'].mean():.2f} / 5" if "rating" in df.columns else "N/A")
    c4.metric("🏆 Top-K", f"{n_topk}")
    c5.metric("🏪 Boutiques", f"{n_shops}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 Produits par Catégorie")
        if "category" in df.columns:
            cc = df["category"].value_counts().head(10)
            fig = px.bar(x=cc.values, y=cc.index, orientation="h",
                         color=cc.values, color_continuous_scale="Blues",
                         labels={"x": "Nb Produits", "y": "Catégorie"})
            fig.update_layout(showlegend=False, height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune information de catégorie disponible.")

    with col2:
        st.subheader("💰 Distribution des Prix par Catégorie")
        if "category" in df.columns and "price" in df.columns:
            top5 = df["category"].value_counts().head(5).index.tolist()
            fig2 = px.box(df[df["category"].isin(top5)], x="category", y="price", color="category",
                          labels={"price": "Prix", "category": "Catégorie"})
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Données insuffisantes pour la distribution des prix.")

    st.markdown("---")
    st.subheader("💡 Insights Automatiques")
    n_out = int((df["dbscan_cluster"] == -1).sum()) if "dbscan_cluster" in df.columns else 0
    pct_stk = (df["is_in_stock_flag"] == 1).mean() * 100 if "is_in_stock_flag" in df.columns else 0
    avg_topk = df[df["is_top_product"] == 1]["price"].mean() if ("price" in df.columns and "is_top_product" in df.columns and (df["is_top_product"] == 1).any()) else 0
    i1, i2, i3, i4 = st.columns(4)
    i1.info(f"🏆 **{n_topk}** Top-K ({n_topk/len(df)*100:.1f}%)")
    i2.warning(f"🚨 **{n_out}** outliers DBSCAN ({n_out/len(df)*100:.1f}%)")
    i3.success(f"📦 **{pct_stk:.0f}%** en stock")
    i4.info(f"🎯 Prix moyen Top-K : **{avg_topk:.2f}**")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Top-K Produits
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Top-K Produits":
    st.title("🏆 Top-K Produits — Classement ML")
    top_k_df = data["topk"].copy() if data.get("topk") is not None else df.head(20)
    if selected_cats and "category" in top_k_df.columns:
        top_k_df = top_k_df[top_k_df["category"].isin(selected_cats)]

    st.markdown(f"**{len(top_k_df)} meilleurs produits** classés par score composite ML")
    cs = [c for c in ["title", "price", "rating", "review_count", "composite_score",
                       "category", "shop_name", "source_platform"] if c in top_k_df.columns]
    st.dataframe(
        top_k_df[cs].sort_values("composite_score", ascending=False) if "composite_score" in top_k_df.columns else top_k_df[cs],
        use_container_width=True, hide_index=True
    )
    if "price" in top_k_df.columns and "rating" in top_k_df.columns:
        st.subheader("💹 Prix vs Note — Top-K")
        cc_col = "category" if "category" in top_k_df.columns else "source_platform"
        fig = px.scatter(top_k_df, x="price", y="rating",
                         size="composite_score" if "composite_score" in top_k_df.columns else None,
                         color=cc_col, hover_data=["title"], size_max=20,
                         labels={"price": "Prix", "rating": "Note /5"})
        st.plotly_chart(fig, use_container_width=True)
    csv = top_k_df[cs].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter CSV", csv, "top_k_products.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Shops & Géo
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏪 Shops & Géo":
    st.title("🏪 Analyse par Boutique & Géographie")
    if "shop_name" not in df.columns:
        st.info("Données boutique non disponibles.")
        st.stop()

    ss = df.groupby("shop_name").agg(
        Nb_produits=("title", "count"),
        Score_moyen=("composite_score", "mean") if "composite_score" in df.columns else ("price", "count"),
        Prix_moyen=("price", "mean"),
        Note_moyenne=("rating", "mean") if "rating" in df.columns else ("price", "count"),
    ).round(3).sort_values("Score_moyen", ascending=False).reset_index()
    ss.columns = ["Boutique", "Nb Produits", "Score Moyen", "Prix Moyen", "Note Moyenne"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Score Moyen par Boutique")
        fig = px.bar(ss.head(15), x="Score Moyen", y="Boutique", orientation="h",
                     color="Score Moyen", color_continuous_scale="Viridis")
        fig.update_layout(height=400, yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📦 Volume par Boutique")
        fig2 = px.pie(ss.head(10), values="Nb Produits", names="Boutique",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Tableau Récapitulatif")
    st.dataframe(ss, use_container_width=True, hide_index=True)

    if "source_platform" in df.columns:
        st.markdown("---")
        st.subheader("🆚 Shopify vs WooCommerce")
        plat = df.groupby("source_platform").agg(
            Produits=("title", "count"),
            Prix_moyen=("price", "mean"),
            Note_moyenne=("rating", "mean") if "rating" in df.columns else ("price", "count"),
            Score_moyen=("composite_score", "mean") if "composite_score" in df.columns else ("price", "count"),
        ).round(3).reset_index()
        st.dataframe(plat, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Clustering & PCA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔵 Clustering & PCA":
    st.title("🔵 Segmentation Produits — KMeans & DBSCAN")
    pca_df = data.get("pca")

    if pca_df is not None and "PC1" in pca_df.columns:
        variance = float(pca_df["pca_variance_explained"].iloc[0]) if "pca_variance_explained" in pca_df.columns else 0
        n_k = pca_df["cluster"].nunique() if "cluster" in pca_df.columns else "?"
        col1, col2 = st.columns(2)
        col1.metric("Variance expliquée PCA (2D)", f"{variance:.1f}%")
        col2.metric("Nombre de clusters", n_k)
        st.subheader(f"📈 PCA 2D — {n_k} clusters (Variance = {variance:.1f}%)")
        fig = _build_pca_scatter(pca_df)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("PCA non disponible — relance `python run_local.py`.")

    if "cluster" in df.columns:
        st.markdown("---")
        st.subheader("📊 Profil des Clusters")
        cc2 = "cluster_label" if "cluster_label" in df.columns else "cluster"
        fc = [c for c in ["price", "rating", "composite_score", "discount_score", "stock"] if c in df.columns]
        summary = df.groupby(cc2)[fc + ["title"]].agg(
            {**{f: "mean" for f in fc}, "title": "count"}
        ).round(3).rename(columns={"title": "Nb Produits"})
        st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Alertes Fraudes & Anomalies")
    suspects = _anomaly_table(pca_df if pca_df is not None else df)
    total_suspicious = len(suspects)
    rate = (total_suspicious / len(pca_df) * 100) if pca_df is not None and len(pca_df) else 0.0
    m1, m2 = st.columns(2)
    m1.metric("Produits suspects", f"{total_suspicious}")
    m2.metric("Taux d'anomalie", f"{rate:.1f}%")

    if not suspects.empty:
        cols = [c for c in [
            "title", "price", "rating", "review_count", "cluster_label",
            "dbscan_outlier", "iforest_outlier", "price_anomaly_score",
            "is_anomaly", "shop_name", "source_platform"
        ] if c in suspects.columns]
        suspects_view = suspects.copy()
        suspects_view["detector_model"] = _anomaly_detector_series(suspects_view)
        st.dataframe(suspects_view[cols + ["detector_model"]], use_container_width=True, hide_index=True)
    else:
        st.success("✅ Aucun produit suspect détecté.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Règles d'Association
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔗 Règles d'Association":
    st.title("🔗 Règles d'Association (Apriori)")
    rules = data.get("rules")
    if rules is None or rules.empty:
        st.warning("Aucune règle générée — le catalogue actuel n'a pas assez de boutiques multi-catégories.")
        st.info(
            "**Pourquoi ?** L'algorithme Apriori nécessite des boutiques vendant **au moins 2 catégories différentes**. "
            "Les boutiques Shopify réelles sont souvent mono-catégorie.\n\n"
            "**Solution** : Génère des données synthétiques multi-boutiques :"
        )
        st.code("python data/generate_synthetic.py 1000\npython run_local.py", language="bash")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Règles extraites", f"{len(rules):,}")
        col2.metric("Lift maximum", f"{rules['lift'].max():.2f}")
        col3.metric("Confiance moyenne", f"{rules['confidence'].mean():.2%}")

        rd = rules.copy()
        rd["antecedents"] = rd["antecedents"].astype(str)
        rd["consequents"] = rd["consequents"].astype(str)
        rd = rd.sort_values("lift", ascending=False).head(30)
        st.subheader("🏆 Top 30 Règles par Lift")
        st.dataframe(rd[["antecedents", "consequents", "support", "confidence", "lift"]],
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Assistant LLM (CORRIGÉ)
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Assistant LLM":
    st.title("🤖 Assistant IA — Analyse e-Commerce")
    st.caption("Intelligence augmentée — fallback automatique Groq → Claude → OpenAI → Gemini")

    # ── Contexte catalogue injecté dans le system prompt ─────────────────────
    _n_topk = int(df["is_top_product"].sum()) if "is_top_product" in df.columns else 0
    _n_out  = int((df["dbscan_cluster"] == -1).sum()) if "dbscan_cluster" in df.columns else 0
    _top_cats = df["category"].value_counts().head(3).index.tolist() if "category" in df.columns else []
    _n_clusters = df["cluster"].nunique() if "cluster" in df.columns else "N/A"
    _avg_score = df["composite_score"].mean() if "composite_score" in df.columns else "N/A"
    _price_min = df["price"].min() if "price" in df.columns else "N/A"
    _price_max = df["price"].max() if "price" in df.columns else "N/A"

    SYSTEM = f"""Tu es un analyste senior en e-commerce et Data Science.
Tu as accès aux données réelles du catalogue via le protocole MCP (Model Context Protocol d'Anthropic).

=== DONNÉES CATALOGUE (temps réel) ===
- Total produits filtrés : {len(df):,} (sur {len(df_full):,} total)
- Prix : min={_price_min:.2f} | max={_price_max:.2f} | moy={df['price'].mean():.2f}
- Note moyenne : {df['rating'].mean():.2f}/5
- Top catégories : {', '.join(_top_cats) if _top_cats else 'N/A'}
- Produits Top-K ML : {_n_topk} ({_n_topk/len(df)*100:.1f}%)
- Outliers DBSCAN : {_n_out} ({_n_out/len(df)*100:.1f}%)
- Clusters KMeans : {_n_clusters}
- Score composite moyen : {_avg_score if isinstance(_avg_score, str) else f'{_avg_score:.3f}'}

=== ARCHITECTURE ML DU PROJET ===
- Scraping A2A : agents Shopify + WooCommerce (API-first + fallback Playwright)
- Feature Engineering : 20 features (prix, popularité, stock, remise, complétude...)
- Modèles supervisés : RandomForest + XGBoost (cible : is_top_product)
- Clustering : KMeans (segments métier) + DBSCAN (outliers) + IsolationForest (anomalies prix)
- Data Mining : règles d'association Apriori (catégories par boutique)
- Visualisation : PCA 2D (réduction dimensionnelle)

=== PRINCIPES MCP (Model Context Protocol Anthropic) ===
- Responsabilité : tes réponses sont basées sur des données réelles, pas des suppositions
- Transparence : cite les métriques ML quand tu analyses
- Isolation : n'expose que les données nécessaires à la réponse

Réponds TOUJOURS en français. Sois concis, précis, orienté décision business.
Utilise les données du catalogue pour ancrer tes réponses dans la réalité.
"""

    # ── Statut des fournisseurs (liste statique, car detect_active_providers supprimée) ──
    supported_providers = ["groq", "anthropic", "openai", "gemini"]
    st.success(f"✅ Fournisseurs supportés : **{' → '.join(supported_providers)}** *(fallback automatique)*")
    st.info(
        "**Configuration recommandée** — Ajoute dans `.env` :\n"
        "```\nGROQ_API_KEY=gsk_...      # ⚡ Recommandé — gratuit & ultra-rapide\n"
        "ANTHROPIC_API_KEY=sk-ant-...\nGEMINI_API_KEY=AIza...    # Gratuit\n```"
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if "llm_msgs" not in st.session_state:
        st.session_state.llm_msgs = []
    if "pending_llm_prompt" not in st.session_state:
        st.session_state.pending_llm_prompt = None

    def _run_llm(user_prompt: str) -> str:
        """Appelle le routeur LLM avec un prompt unique construit à partir de l'historique et du système."""
        # Construction de l'historique complet sous forme de texte pour le modèle
        history_text = ""
        for msg in st.session_state.llm_msgs:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        # On ajoute le nouveau message utilisateur
        full_prompt = f"{SYSTEM}\n\nHistorique de la conversation :\n{history_text}\nUtilisateur: {user_prompt}\nAssistant:"
        # Appel au routeur
        return generate_response(
            prompt=full_prompt,
            provider="groq",         # premier essai
            fallback_enabled=True,   # fallback automatique
            temperature=0.3,
            max_tokens=1200
        )

    # ── Traitement du prompt en attente (depuis boutons suggestion) ───────────
    if st.session_state.pending_llm_prompt is not None:
        pending = st.session_state.pending_llm_prompt
        st.session_state.pending_llm_prompt = None
        st.session_state.llm_msgs.append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                reply = _run_llm(pending)
            st.markdown(reply)
        st.session_state.llm_msgs.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Affichage historique ──────────────────────────────────────────────────
    for msg in st.session_state.llm_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Saisie manuelle ───────────────────────────────────────────────────────
    if prompt := st.chat_input("Ex: Analyse les opportunités dans la catégorie Sport"):
        st.session_state.llm_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                reply = _run_llm(prompt)
            st.markdown(reply)
        st.session_state.llm_msgs.append({"role": "assistant", "content": reply})

    # ── Bouton effacer ────────────────────────────────────────────────────────
    if st.session_state.llm_msgs:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.llm_msgs = []
            st.rerun()

    # ── Suggestions rapides ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**💡 Questions suggérées :**")
    suggs = [
        "Résume les insights business du catalogue",
        "Quels clusters méritent une stratégie premium ?",
        "Analyse les anomalies de prix détectées",
        "Recommande une stratégie de pricing",
        "Quelles catégories ont le meilleur rapport qualité/prix ?",
    ]
    cols = st.columns(len(suggs))
    for col, q in zip(cols, suggs):
        if col.button(q, use_container_width=True, key=f"sug_{hash(q)}"):
            st.session_state.pending_llm_prompt = q
            st.rerun()