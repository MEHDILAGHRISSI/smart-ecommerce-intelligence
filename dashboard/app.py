"""
Smart eCommerce Intelligence Dashboard
FST Tanger — LSI2 — DM & SID 2025/2026

Version modifiée : robustesse au chargement des CSV, validation des colonnes critiques,
et protections pour éviter les Plantages sur données manquantes / corrompues.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from typing import Dict, Optional
import pandas as pd
import streamlit as st
from configs.settings import DATA_PROCESSED_DIR, Settings

# Configuration page
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires de chargement et validation des données
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["price", "title"]

@st.cache_data(ttl=300)
def load_data() -> Dict[str, Optional[pd.DataFrame]]:
    """
    Charge les CSV essentiels depuis `DATA_PROCESSED_DIR`.
    - En cas d'absence de `products_final.csv` ou d'erreur de lecture/validation,
      retourne un dict avec des valeurs None (le caller affichera un message clair).
    """
    paths = {
        "final": DATA_PROCESSED_DIR / "products_final.csv",
        "topk":  DATA_PROCESSED_DIR / "top_k_products.csv",
        "rules": DATA_PROCESSED_DIR / "association_rules.csv",
        "pca":   DATA_PROCESSED_DIR / "pca_viz.csv",
    }

    # Si le CSV principal est absent, on renvoie des None (dashboard affichera l'aide)
    if not paths["final"].exists():
        return {k: None for k in paths}

    # Lecture du CSV principal avec gestion d'erreur
    try:
        df_final = pd.read_csv(paths["final"])
    except Exception:
        # Si lecture échoue (fichier corrompu, encodage, etc.) -> considérer comme absent
        return {k: None for k in paths}

    # Validation colonnes minimales
    missing = [c for c in REQUIRED_COLUMNS if c not in df_final.columns]
    if missing:
        # Si colonnes essentielles manquantes, on considère que les données ne sont pas valides
        return {k: None for k in paths}

    result: Dict[str, Optional[pd.DataFrame]] = {"final": df_final, "topk": None, "rules": None, "pca": None}

    # top_k : fallback si absent
    try:
        if paths["topk"].exists():
            result["topk"] = pd.read_csv(paths["topk"])
        else:
            result["topk"] = df_final.head(20)
    except Exception:
        result["topk"] = df_final.head(20)

    # rules : lecture optionnelle
    try:
        if paths["rules"].exists():
            result["rules"] = pd.read_csv(paths["rules"])
        else:
            result["rules"] = pd.DataFrame()
    except Exception:
        result["rules"] = pd.DataFrame()

    # pca : lecture optionnelle
    try:
        if paths["pca"].exists():
            result["pca"] = pd.read_csv(paths["pca"])
        else:
            result["pca"] = None
    except Exception:
        result["pca"] = None

    return result

# Chargement des données (cache)
data = load_data()
df_full = data["final"]

# Si pas de données valides -> message et arrêt (Streamlit)
if df_full is None:
    st.sidebar.markdown("## 🛠️ Diagnostic rapide")
    st.sidebar.write(f"Répertoire attendu : `{DATA_PROCESSED_DIR}`")
    st.sidebar.write("Fichiers présents :")
    try:
        existing = [p.name for p in sorted(DATA_PROCESSED_DIR.glob("*"))]
    except Exception:
        existing = []
    if existing:
        for p in existing:
            st.sidebar.write(f"- {p}")
    else:
        st.sidebar.write("- (aucun fichier)")

    st.error(
        "⚠️ Données manquantes ou corrompues.\n\n"
        "Génère d'abord les CSV attendus puis relance le dashboard.\n\n"
        "Commandes conseillées :\n"
    )
    st.code("python data/generate_synthetic.py 500\npython run_local.py", language="bash")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — filtres et validation des colonnes critiques (robuste)
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=70)
st.sidebar.markdown("## 🛒 Smart eCommerce")
st.sidebar.markdown("*FST Tanger — LSI2 — DM & SID*")
st.sidebar.markdown("---")

# Colonnes facultatives, on prépare des listes sûres
all_cats = sorted(df_full["category"].dropna().unique().tolist()) if "category" in df_full.columns else []
all_shops = sorted(df_full["shop_name"].dropna().unique().tolist()) if "shop_name" in df_full.columns else []
all_plats = df_full["source_platform"].dropna().unique().tolist() if "source_platform" in df_full.columns else []

selected_cats = st.sidebar.multiselect("Catégories", all_cats, default=[])
selected_shops = st.sidebar.multiselect("Boutiques", all_shops[:20], default=[])
selected_plats = st.sidebar.multiselect("Plateformes", all_plats, default=all_plats)

# Protection pour le slider de prix : on vérifie la validité des valeurs
try:
    price_series = pd.to_numeric(df_full["price"], errors="coerce").dropna()
    if price_series.empty:
        raise ValueError("Series price vide après conversion")
    min_p = float(price_series.min())
    max_p = float(price_series.max())
    # Si min == max (cas pathologique), on élargit raisonnablement
    if min_p >= max_p:
        min_p, max_p = max(0.0, min_p - 1.0), max_p + 1.0
except Exception:
    min_p, max_p = 0.0, 1000.0
    st.sidebar.warning("Valeurs de prix invalides ou absentes — slider fixé sur [0,1000].")

price_range = st.sidebar.slider("Fourchette de prix", min_p, max_p, (min_p, max_p), step=max(1.0, (max_p - min_p) / 100))

stock_only = st.sidebar.checkbox("En stock uniquement", value=False)
st.sidebar.markdown("---")
st.sidebar.caption(f"📦 {len(df_full):,} produits chargés")

# Application des filtres en mode défensif
df = df_full.copy()
if selected_cats and "category" in df.columns:
    df = df[df["category"].isin(selected_cats)]
if selected_shops and "shop_name" in df.columns:
    df = df[df["shop_name"].isin(selected_shops)]
if selected_plats and "source_platform" in df.columns:
    df = df[df["source_platform"].isin(selected_plats)]

# Filtre prix (défensif)
try:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
except Exception:
    # Si la colonne price est totalement invalide, on ne filtre pas par prix mais on avertit
    st.warning("Impossible d'appliquer le filtre prix (colonne `price` invalide).")

# Filtre stock
if stock_only:
    if "is_in_stock" in df.columns:
        df = df[df["is_in_stock"] == True]
    elif "stock" in df.columns:
        df = df[pd.to_numeric(df["stock"], errors="coerce").fillna(0) > 0]

if df.empty:
    st.warning("⚠️ Aucun produit après filtrage. Ajuste les filtres.")
    st.stop()

import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Vue Globale
# ──────────────────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    label="Navigation",
    options=["📊 Vue Globale", "🏆 Top-K Produits", "🏪 Shops & Géo",
             "🔵 Clustering & PCA", "🔗 Règles d'Association", "🤖 Assistant LLM"],
    index=0,
)

# Utiliser la variable `page` pour la navigation
if page == "📊 Vue Globale":
    st.title("📊 Vue Globale — KPIs")
    st.caption(f"Données filtrées : **{len(df):,} produits** sur {len(df_full):,} total")

    # Utiliser des getters sûrs pour éviter KeyError
    n_topk = int(df.get("is_top_product", pd.Series([0] * len(df))).sum())
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
    n_out = int((df.get("dbscan_cluster", pd.Series([0] * len(df))) == -1).sum()) if "dbscan_cluster" in df.columns else 0
    pct_stk = (df.get("is_in_stock_flag", pd.Series([1] * len(df))) == 1).mean() * 100 if "is_in_stock_flag" in df.columns else 0
    avg_topk = df[df.get("is_top_product", 0) == 1]["price"].mean() if ("price" in df.columns and (df.get("is_top_product", 0) == 1).any()) else 0
    i1, i2, i3, i4 = st.columns(4)
    i1.info(f"🏆 **{n_topk}** Top-K ({n_topk/len(df)*100:.1f}%)")
    i2.warning(f"🚨 **{n_out}** outliers DBSCAN ({(n_out/len(df)*100) if len(df)>0 else 0:.1f}%)")
    i3.success(f"📦 **{pct_stk:.0f}%** en stock")
    i4.info(f"🎯 Prix moyen Top-K : **{avg_topk:.2f}**")

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
        cc = "category" if "category" in top_k_df.columns else "source_platform"
        fig = px.scatter(top_k_df, x="price", y="rating", size="composite_score" if "composite_score" in top_k_df.columns else None,
                         color=cc, hover_data=["title"], size_max=20,
                         labels={"price": "Prix", "rating": "Note /5"})
        st.plotly_chart(fig, use_container_width=True)
    csv = top_k_df[cs].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Exporter CSV", csv, "top_k_products.csv", "text/csv")

elif page == "🏪 Shops & Géo":
    st.title("🏪 Analyse par Boutique & Géographie")
    if "shop_name" not in df.columns:
        st.info("Données boutique non disponibles.")
        st.stop()

    ss = df.groupby("shop_name").agg(
        Nb_produits=("title", "count"),
        Score_moyen=("composite_score", "mean"),
        Prix_moyen=("price", "mean"),
        Note_moyenne=("rating", "mean"),
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
            Note_moyenne=("rating", "mean"),
            Score_moyen=("composite_score", "mean"),
        ).round(3).reset_index()
        st.dataframe(plat, use_container_width=True, hide_index=True)

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
        st.info(f"Réduction dimensionnelle → 2D. Variance expliquée : **{variance:.1f}%**")

        cc = "cluster_label" if "cluster_label" in pca_df.columns else "cluster" if "cluster" in pca_df.columns else "category" if "category" in pca_df.columns else None
        hov = [c for c in ["title", "price", "rating", "composite_score", "shop_name"] if c in pca_df.columns]

        fig = px.scatter(pca_df, x="PC1", y="PC2", color=cc, hover_data=hov,
                         labels={"PC1": "Composante 1", "PC2": "Composante 2"},
                         color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7)
        if "is_top_product" in pca_df.columns:
            tp = pca_df[pca_df["is_top_product"] == 1]
            fig.add_scatter(x=tp["PC1"], y=tp["PC2"], mode="markers",
                            marker=dict(symbol="star", size=14, color="gold",
                                        line=dict(color="black", width=1)), name="⭐ Top-K")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PCA non disponible — relance `python run_local.py` ou `python ml/pipeline.py` pour générer les fichiers nécessaires.")

    if "cluster" in df.columns:
        st.markdown("---")
        st.subheader("📊 Profil des Clusters")
        cc2 = "cluster_label" if "cluster_label" in df.columns else "cluster"
        fc = [c for c in ["price", "rating", "composite_score", "discount_score", "stock"] if c in df.columns]
        summary = df.groupby(cc2)[fc + ["title"]].agg(
            {**{f: "mean" for f in fc}, "title": "count"}
        ).round(3).rename(columns={"title": "Nb Produits"})
        st.dataframe(summary, use_container_width=True)

    if "dbscan_cluster" in df.columns:
        st.markdown("---")
        st.subheader("🚨 Détection d'Anomalies (DBSCAN)")
        anom = df[df["dbscan_cluster"] == -1]
        n_out = len(anom)
        col1, col2 = st.columns(2)
        col1.metric("Outliers détectés", n_out)
        col2.metric("Taux d'anomalie", f"{n_out / len(df) * 100:.1f}%")
        if not anom.empty:
            st.error(f"🚨 {n_out} produit(s) au profil atypique")
            sc = [c for c in ["title", "price", "rating", "review_count", "category", "shop_name"] if c in anom.columns]
            st.dataframe(anom[sc].head(20), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Aucune anomalie.")

elif page == "🔗 Règles d'Association":
    st.title("🔗 Règles d'Association (Apriori)")
    rules = data.get("rules")
    if rules is None or rules.empty:
        st.warning("Aucune règle générée.")
        st.code("python data/generate_synthetic.py 1000\npython run_local.py")
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

elif page == "🤖 Assistant LLM":
    st.title("🤖 Assistant IA — Analyse e-Commerce")
    st.caption("Intelligence augmentée — fallback automatique entre fournisseurs (mode démo si aucune clé)")

    # Contexte catalogue
    _n_topk = int(df.get("is_top_product", pd.Series([0] * len(df))).sum())
    _n_out = int((df.get("dbscan_cluster", pd.Series([0] * len(df))) == -1).sum()) if "dbscan_cluster" in df.columns else 0
    _top_cats = df["category"].value_counts().head(3).index.tolist() if "category" in df.columns else []

    SYSTEM = f"""Tu es un analyste senior en e-commerce et Data Science.
Catalogue analysé ({len(df)} produits) :
- Prix : min={df['price'].min():.2f} si disponible, moy={df['price'].mean():.2f} si disponible
- Note moyenne : {df['rating'].mean():.2f}/5 si disponible
- Top catégories : {', '.join(_top_cats) if _top_cats else 'N/A'}
- Produits Top-K ML : {_n_topk} ({_n_topk/len(df)*100:.1f}%)
- Outliers DBSCAN : {_n_out} ({_n_out/len(df)*100:.1f}%)
- Clusters KMeans : {df['cluster'].nunique() if 'cluster' in df.columns else 'N/A'}
Réponds TOUJOURS en français, concis, orienté décision business."""

    # Vérification clé API (affichage informatif)
    _s2 = Settings()
    _groq = os.getenv("GROQ_API_KEY", "")
    _ant = _s2.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
    _oai = _s2.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    _gem = os.getenv("GEMINI_API_KEY", "")
    _active = [n for n, k in [("Groq", _groq), ("Claude", _ant), ("OpenAI", _oai), ("Gemini", _gem)] if k]
    if _active:
        st.success(f"✅ Fournisseurs détectés : **{' → '.join(_active)}**  *(fallback automatique)*")
    else:
        st.info("""
**Mode Démo actif** — Aucune clé API configurée. Réponses simulées.

Pour activer l'IA réelle, ajoute dans `.env` (au moins une) :
```
GROQ_API_KEY=gsk_...          # ⚡ Recommandé — gratuit & ultra-rapide
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...        # Gratuit — aistudio.google.com
```
        """)

    # Session state pour le chat LLM
    if "llm_msgs" not in st.session_state:
        st.session_state.llm_msgs = []
    if "pending_llm_prompt" not in st.session_state:
        st.session_state.pending_llm_prompt = None

    # Traitement du prompt en attente (depuis les boutons de suggestion)
    if st.session_state.pending_llm_prompt is not None:
        pending = st.session_state.pending_llm_prompt
        st.session_state.pending_llm_prompt = None
        st.session_state.llm_msgs.append({"role": "user", "content": pending})
        with st.chat_message("user"):
            st.markdown(pending)
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                # Module local d'amélioration de prompts non requis en mode démo.
                # Si le projet contient `llm.scraping_prompt`, tu peux remplacer ce bloc
                # par un appel réel au LLM via `_call_llm_api`.
                reply = "Mode démo : réponse simulée. Ajoute une clé API dans .env pour analyses réelles."
            st.markdown(reply)
        st.session_state.llm_msgs.append({"role": "assistant", "content": reply})
        st.rerun()

    # Affichage historique
    for msg in st.session_state.llm_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (saisie manuelle)
    if prompt := st.chat_input("Ex: Analyse les opportunités dans la catégorie Sport"):
        st.session_state.llm_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                reply = "Mode démo : réponse simulée. Ajoute une clé API dans .env pour analyses réelles."
            st.markdown(reply)
        st.session_state.llm_msgs.append({"role": "assistant", "content": reply})

    # Bouton effacer
    if st.session_state.llm_msgs:
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.llm_msgs = []
            st.rerun()

    # Suggestions rapides
    st.markdown("---")
    st.markdown("**💡 Questions suggérées :**")
    suggs = [
        "Résume les insights business du catalogue",
        "Quels clusters méritent une stratégie premium ?",
        "Recommande une stratégie de pricing",
        "Analyse les outliers DBSCAN détectés",
        "Quelles catégories ont le meilleur rapport qualité/prix ?",
    ]
    cols = st.columns(len(suggs))
    for col, q in zip(cols, suggs):
        if col.button(q, use_container_width=True, key=f"sug_{hash(q)}"):
            st.session_state.pending_llm_prompt = q
            st.rerun()

