"""Composant pour la visualisation du clustering (KMeans & DBSCAN)."""

import streamlit as st
import pandas as pd
import plotly.express as px


def show_cluster_chart(df: pd.DataFrame) -> None:
    """Affiche le scatter plot des clusters (PCA ou Prix/Note) et les anomalies."""
    if "cluster" not in df.columns:
        st.warning("⚠️ Les données de clustering KMeans ne sont pas disponibles.")
        return

    st.write("### Segmentation des produits (KMeans)")

    # Copie locale et conversion du cluster en chaîne (catégorie discrète pour Plotly)
    plot_df = df.copy()
    plot_df["cluster_label"] = "Segment " + plot_df["cluster"].astype(str)

    # Définition de la taille des bulles
    size_col = "review_count" if "review_count" in plot_df.columns and plot_df["review_count"].sum() > 0 else None
    if size_col:
        # Éviter les tailles à 0 pour Plotly
        plot_df[size_col] = plot_df[size_col].fillna(0) + 1

    # Vérification de la présence des colonnes PCA
    use_pca = ("pca_1" in plot_df.columns) and ("pca_2" in plot_df.columns)

    if use_pca:
        x_col, y_col = "pca_1", "pca_2"
        x_label = "Composante Principale 1"
        y_label = "Composante Principale 2"
        explanation = (
            "Ce graphique représente les produits dans un espace réduit à 2 dimensions par "
            "**Analyse en Composantes Principales (PCA)** à partir des features ML "
            "(prix, popularité, densité d'avis, etc.). "
            "Les clusters sont bien séparés : la proximité visuelle reflète la similarité "
            "multidimensionnelle des produits."
        )
    else:
        x_col, y_col = "price", "rating"
        x_label = "Prix"
        y_label = "Note / 5"
        explanation = (
            "Ce graphique croise le Prix et la Note. "
            "La taille des bulles représente le nombre d'avis."
        )

    # Ajout des informations métier au survol
    hover_cols = ["title", "price", "rating"]
    if "source_platform" in plot_df.columns:
        hover_cols.append("source_platform")
    if "shop_name" in plot_df.columns:
        hover_cols.append("shop_name")

    st.info(explanation)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="cluster_label",
        size=size_col,
        hover_data=hover_cols,
        labels={x_col: x_label, y_col: y_label, "cluster_label": "Cluster"},
        template="plotly_white"
    )

    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
    st.plotly_chart(fig, use_container_width=True)

    # Section DBSCAN (Anomalies)
    if "dbscan_cluster" in df.columns:
        st.markdown("---")
        st.write("### Détection d'anomalies (DBSCAN)")
        anomalies = df[df["dbscan_cluster"] == -1]

        if not anomalies.empty:
            st.error(f"🚨 {len(anomalies)} produit(s) au profil atypique (outliers) détecté(s).")
            st.dataframe(
                anomalies[["title", "price", "rating", "review_count", "source_platform"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("✅ Aucune anomalie majeure détectée dans le catalogue.")