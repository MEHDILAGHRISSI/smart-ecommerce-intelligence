"""Composant pour la visualisation du clustering (KMeans & DBSCAN)."""

import streamlit as st
import pandas as pd
import plotly.express as px


def show_cluster_chart(df: pd.DataFrame) -> None:
    """Affiche le scatter plot des clusters et les anomalies."""
    if "cluster" not in df.columns:
        st.warning("⚠️ Les données de clustering KMeans ne sont pas disponibles.")
        return

    st.write("### Segmentation des produits (KMeans)")
    st.info("Ce graphique croise le Prix et la Note. La taille des bulles représente le nombre d'avis.")

    # Copie locale et conversion du cluster en chaîne (catégorie discrète pour Plotly)
    plot_df = df.copy()
    plot_df["cluster_label"] = "Segment " + plot_df["cluster"].astype(str)

    # Définition de la taille des bulles
    size_col = "review_count" if "review_count" in plot_df.columns and plot_df["review_count"].sum() > 0 else None
    if size_col:
        # Éviter les tailles à 0 pour Plotly
        plot_df[size_col] = plot_df[size_col].fillna(0) + 1

    fig = px.scatter(
        plot_df,
        x="price",
        y="rating",
        color="cluster_label",
        size=size_col,
        hover_data=["title", "source_platform"],
        labels={"price": "Prix", "rating": "Note / 5", "cluster_label": "Cluster"},
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