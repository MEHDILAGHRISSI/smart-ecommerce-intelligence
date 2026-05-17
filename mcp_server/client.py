"""
Smart eCommerce Intelligence — MCP Client
FST Tanger — LSI2 — DM & SID 2025/2026

Client MCP à utiliser depuis le dashboard Streamlit ou le LLM.
Permet à l'assistant IA d'appeler les outils e-commerce de façon responsable.

Usage dans dashboard/app.py :
    from mcp_server.client import MCPClient
    mcp = MCPClient()
    top_products = mcp.get_top_products(limit=10, category="Électronique")

Usage avec Claude API (A2A) :
    from mcp_server.client import MCPClient, build_mcp_system_prompt
    mcp = MCPClient()
    prompt = build_mcp_system_prompt(mcp)
"""

from __future__ import annotations
import json
import os
import requests
from typing import Any

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")


class MCPClient:
    """
    Client MCP léger pour le dashboard Streamlit.

    Mode local : appelle directement le serveur Python sans HTTP.
    Mode HTTP  : appelle le serveur MCP via HTTP REST.
    """

    def __init__(self, server_url: str = None, local_mode: bool = True):
        self.server_url = server_url or MCP_SERVER_URL
        self.local_mode = local_mode

        if local_mode:
            # Import direct du serveur (plus rapide, pas de réseau)
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from mcp_server.server import SmartEcommerceMCPServer
                self._server = SmartEcommerceMCPServer()
            except Exception as e:
                print(f"⚠️ MCP local non disponible: {e}. Mode HTTP activé.")
                self.local_mode = False
                self._server = None

    def _call(self, tool_name: str, **kwargs) -> dict:
        """Appel interne — local ou HTTP selon le mode."""
        # Filtrer les arguments None
        args = {k: v for k, v in kwargs.items() if v is not None}

        if self.local_mode and self._server:
            return self._server.call_tool(tool_name, args)
        else:
            try:
                resp = requests.post(
                    f"{self.server_url}/tools/{tool_name}",
                    json=args,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                return {"error": f"MCP HTTP error: {e}"}

    # ── API publique ──────────────────────────────────────────────────────────

    def get_top_products(self, limit: int = 10, category: str = None,
                         min_score: float = None) -> dict:
        """Top-K produits classés par ML."""
        return self._call("get_top_products", limit=limit,
                          category=category, min_score=min_score)

    def get_cluster_summary(self, include_products: bool = False) -> dict:
        """Profil des clusters KMeans."""
        return self._call("get_cluster_summary", include_products=include_products)

    def get_anomalies(self, limit: int = 20) -> dict:
        """Anomalies détectées par DBSCAN."""
        return self._call("get_anomalies", limit=limit)

    def get_association_rules(self, min_lift: float = 1.0, limit: int = 10) -> dict:
        """Règles d'association Apriori."""
        return self._call("get_association_rules", min_lift=min_lift, limit=limit)

    def get_catalog_stats(self, group_by: str = "category") -> dict:
        """Statistiques globales du catalogue."""
        return self._call("get_catalog_stats", group_by=group_by)

    def search_products(self, query: str = None, category: str = None,
                        min_price: float = None, max_price: float = None,
                        min_rating: float = None, in_stock_only: bool = False,
                        limit: int = 20) -> dict:
        """Recherche multi-critères."""
        return self._call("search_products", query=query, category=category,
                          min_price=min_price, max_price=max_price,
                          min_rating=min_rating, in_stock_only=in_stock_only,
                          limit=limit)

    def get_shop_ranking(self, limit: int = 10, sort_by: str = "composite_score") -> dict:
        """Classement des boutiques."""
        return self._call("get_shop_ranking", limit=limit, sort_by=sort_by)


def build_mcp_system_prompt(mcp: MCPClient) -> str:
    """
    Construit le system prompt MCP pour le LLM.
    Injecte les données réelles du catalogue dans le contexte de l'IA.

    C'est ici qu'on applique le principe MCP de contextualisation :
    le LLM reçoit exactement ce dont il a besoin, rien de plus.
    """
    stats = mcp.get_catalog_stats()
    top = mcp.get_top_products(limit=5)
    anomalies = mcp.get_anomalies(limit=3)
    clusters = mcp.get_cluster_summary(include_products=True)

    g = stats.get("global", {})
    top_products_str = "\n".join([
        f"  - {p.get('title', '?')[:40]} | Prix: {p.get('price', '?')} | Score: {p.get('composite_score', '?')}"
        for p in top.get("products", [])[:5]
    ]) or "  (données non disponibles)"

    cluster_str = "\n".join([
        f"  - Segment {c.get('cluster_id')}: {c.get('segment_type', '?')} | {c.get('n_products', 0)} produits | Score moy: {c.get('composite_score', '?')}"
        for c in clusters.get("clusters", [])[:5]
    ]) or "  (données non disponibles)"

    prompt = f"""Tu es un analyste senior en e-commerce et Data Science.
Tu as accès aux données réelles du catalogue via le protocole MCP (Model Context Protocol d'Anthropic).

=== DONNÉES CATALOGUE (temps réel via MCP) ===
- Total produits : {g.get('total_products', 'N/A')}
- Prix moyen : {g.get('avg_price', 'N/A')}
- Note moyenne : {g.get('avg_rating', 'N/A')} / 5
- Catégories : {g.get('n_categories', 'N/A')}
- Boutiques : {g.get('n_shops', 'N/A')}

=== TOP 5 PRODUITS ML ===
{top_products_str}

=== SEGMENTS KMEANS ===
{cluster_str}

=== ANOMALIES DBSCAN ===
{anomalies.get('total_anomalies', 0)} outliers ({anomalies.get('anomaly_rate_pct', 0):.1f}% du catalogue)

=== PRINCIPES MCP ===
- Responsabilité : tes réponses sont basées sur des données réelles, pas des suppositions
- Transparence : cite les métriques ML quand tu analyses
- Décision : oriente toujours vers des actions business concrètes

Réponds en français, de façon concise et orientée décision business.
"""
    return prompt


# ── Test standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔌 Test MCP Client\n")
    mcp = MCPClient(local_mode=True)

    print("📊 Stats catalogue :")
    print(json.dumps(mcp.get_catalog_stats(), default=str, indent=2, ensure_ascii=False)[:500])

    print("\n🏆 Top 5 produits :")
    top = mcp.get_top_products(limit=5)
    for p in top.get("products", []):
        print(f"  - {p.get('title', '?')[:40]} | Score: {p.get('composite_score', '?')}")

    print("\n🤖 System prompt MCP généré :")
    print(build_mcp_system_prompt(mcp)[:300], "...")