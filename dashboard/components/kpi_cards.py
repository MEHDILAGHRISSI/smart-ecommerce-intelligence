"""Composant pour l'affichage des KPIs globaux."""

import streamlit as st
import pandas as pd


def show_kpis(df: pd.DataFrame) -> None:
    """Affiche une ligne de métriques clés basées sur le dataset."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Produits", len(df))
    with col2:
        st.metric("Prix Moyen", f"{df['price'].mean():.2f} €")
    with col3:
        st.metric("Note Moyenne", f"{df['rating'].mean():.2f} / 5")
    with col4:
        # Gère le cas où review_count pourrait être absent ou nul
        total_reviews = df['review_count'].sum() if 'review_count' in df.columns else 0
        st.metric("Total Avis", f"{total_reviews:,.0f}".replace(",", " "))