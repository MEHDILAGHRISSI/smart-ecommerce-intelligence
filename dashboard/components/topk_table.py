"""Composant pour l'affichage du classement Top-K."""

import streamlit as st
import pandas as pd


def show_topk_table(df: pd.DataFrame) -> None:
    """Affiche un tableau interactif des meilleurs produits."""
    st.write("Voici les produits identifiés comme les plus prometteurs par le pipeline ML :")

    # Sélection des colonnes pertinentes
    cols_to_show = ["title", "price", "rating", "review_count", "composite_score", "source_platform"]
    display_df = df[[c for c in cols_to_show if c in df.columns]].copy()

    if "composite_score" in display_df.columns:
        display_df["composite_score"] = display_df["composite_score"].round(3)

    # Affichage avec formatage Streamlit natif
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "title": "Produit",
            "price": st.column_config.NumberColumn("Prix", format="%.2f €"),
            "rating": st.column_config.NumberColumn("Note", format="%.1f ⭐"),
            "review_count": "Nombre d'avis",
            "composite_score": st.column_config.ProgressColumn(
                "Score Global (ML)",
                format="%f",
                min_value=0.0,
                max_value=1.0,
            ),
            "source_platform": "Plateforme"
        }
    )