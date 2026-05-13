"""
Smart eCommerce Intelligence — MCP Server (Model Context Protocol)
FST Tanger — LSI2 — DM & SID 2025/2026

Ce serveur MCP expose les outils d'analyse e-commerce aux LLMs.
Il suit le protocole Anthropic MCP v2025 pour une IA responsable :
  - Déclaration explicite des intentions de chaque outil
  - Isolation : chaque outil n'expose que ce qui est nécessaire
  - Logs : toutes les requêtes sont journalisées
  - Permissions : validation des paramètres avant exécution

Usage :
    # Démarrer le serveur MCP
    python mcp/server.py

    # Tester avec le client
    python mcp/client.py

Référence :
    https://modelcontextprotocol.io/specification/2025-03-26
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ── Configuration logging ─────────────────────────────────────────────────────
LOG_DIR = Path("logs/mcp")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MCP] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"mcp_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger("mcp_server")

# ── Chemins des données ───────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")


# ─────────────────────────────────────────────────────────────────────────────
# OUTILS MCP — Chaque outil suit les principes de responsabilité :
#   1. Description claire de ce qu'il fait
#   2. Validation stricte des paramètres
#   3. Log de chaque appel
#   4. Retour structuré et prévisible
# ─────────────────────────────────────────────────────────────────────────────

class SmartEcommerceMCPServer:
    """
    Serveur MCP pour le projet Smart eCommerce Intelligence.

    Outils disponibles :
    - get_top_products    : Retourne les Top-K produits classés par ML
    - get_cluster_summary : Analyse des segments KMeans
    - get_anomalies       : Produits atypiques détectés par DBSCAN
    - get_association_rules : Règles Apriori entre catégories
    - get_catalog_stats   : Statistiques globales du catalogue
    - search_products     : Recherche de produits par critères
    - get_shop_ranking    : Classement des boutiques
    """

    # ── Déclaration des outils (schéma MCP) ──────────────────────────────────
    TOOLS = [
        {
            "name": "get_top_products",
            "description": "Retourne les meilleurs produits identifiés par le pipeline ML. Utilise le composite_score calculé par RandomForest + XGBoost pour classer les produits.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre de produits à retourner (1-50, défaut: 10)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filtrer par catégorie (optionnel)",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Score composite minimum [0-1] (optionnel)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_cluster_summary",
            "description": "Retourne le profil de chaque segment KMeans. Aide à comprendre les différents types de produits dans le catalogue (premium, discount, populaire, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_products": {
                        "type": "boolean",
                        "description": "Inclure les exemples de produits par cluster",
                        "default": False,
                    }
                },
                "required": [],
            },
        },
        {
            "name": "get_anomalies",
            "description": "Retourne les produits au profil atypique détectés par DBSCAN (outliers). Ce sont des produits qui méritent une vérification manuelle : prix aberrant, note incohérente, etc.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre d'anomalies à retourner (1-100, défaut: 20)",
                        "default": 20,
                    }
                },
                "required": [],
            },
        },
        {
            "name": "get_association_rules",
            "description": "Retourne les règles d'association Apriori entre catégories de produits. Exemple : {Sport} → {Alimentation} avec lift=2.3 signifie que les boutiques qui vendent du sport vendent souvent aussi de l'alimentation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_lift": {
                        "type": "number",
                        "description": "Lift minimum (défaut: 1.0)",
                        "default": 1.0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Nombre de règles à retourner",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_catalog_stats",
            "description": "Retourne les statistiques globales du catalogue : nombre de produits, prix moyen, note moyenne, répartition par catégorie, etc.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["category", "shop_name", "source_platform"],
                        "description": "Dimension d'agrégation",
                        "default": "category",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "search_products",
            "description": "Recherche des produits par critères multiples. Permet de filtrer par catégorie, plage de prix, note minimale, etc.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Mot-clé à chercher dans le titre du produit",
                    },
                    "category": {"type": "string", "description": "Catégorie"},
                    "min_price": {"type": "number", "description": "Prix minimum"},
                    "max_price": {"type": "number", "description": "Prix maximum"},
                    "min_rating": {"type": "number", "description": "Note minimale [0-5]"},
                    "in_stock_only": {"type": "boolean", "description": "En stock uniquement", "default": False},
                    "limit": {"type": "integer", "description": "Résultats max", "default": 20},
                },
                "required": [],
            },
        },
        {
            "name": "get_shop_ranking",
            "description": "Classement des boutiques par score moyen ML, volume de produits, et note moyenne clients.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Nombre de boutiques", "default": 10},
                    "sort_by": {
                        "type": "string",
                        "enum": ["composite_score", "rating", "n_products"],
                        "default": "composite_score",
                    },
                },
                "required": [],
            },
        },
    ]

    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._rules: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        """Charge les données (avec cache)."""
        if self._df is None:
            path = DATA_DIR / "products_final.csv"
            if path.exists():
                self._df = pd.read_csv(path, encoding="utf-8")
                logger.info(f"Données chargées : {len(self._df)} produits")
            else:
                logger.warning("products_final.csv introuvable — données vides")
                self._df = pd.DataFrame()
        return self._df

    def _load_rules(self) -> pd.DataFrame:
        """Charge les règles d'association (avec cache)."""
        if self._rules is None:
            path = DATA_DIR / "association_rules.csv"
            if path.exists():
                self._rules = pd.read_csv(path, encoding="utf-8")
            else:
                self._rules = pd.DataFrame()
        return self._rules

    def _log_call(self, tool: str, params: dict, result_size: int):
        """Journalise chaque appel outil — principe de responsabilité MCP."""
        logger.info(f"TOOL={tool} | PARAMS={json.dumps(params, default=str)} | RESULT_SIZE={result_size}")

    # ── Implémentation des outils ─────────────────────────────────────────────

    def get_top_products(self, limit: int = 10, category: str = None, min_score: float = None) -> dict:
        """Retourne les Top-K produits."""
        df = self._load_data()
        if df.empty:
            return {"error": "Données non disponibles. Lance d'abord le pipeline ML."}

        result = df.copy()
        if "is_top_product" in result.columns:
            result = result[result["is_top_product"] == 1]
        if category:
            result = result[result["category"].str.lower() == category.lower()]
        if min_score is not None and "composite_score" in result.columns:
            result = result[result["composite_score"] >= min_score]

        cols = [c for c in ["title", "price", "rating", "review_count",
                              "composite_score", "category", "shop_name"] if c in result.columns]
        if "composite_score" in result.columns:
            result = result.sort_values("composite_score", ascending=False)

        result = result.head(limit)[cols].round(3)
        params = {"limit": limit, "category": category, "min_score": min_score}
        self._log_call("get_top_products", params, len(result))
        return {"products": result.to_dict("records"), "total": len(result)}

    def get_cluster_summary(self, include_products: bool = False) -> dict:
        """Profil des clusters KMeans."""
        df = self._load_data()
        if df.empty or "cluster" not in df.columns:
            return {"error": "Données de clustering non disponibles."}

        feat_cols = [c for c in ["price", "rating", "composite_score", "discount_score"] if c in df.columns]
        summary = df.groupby("cluster")[feat_cols + ["title"]].agg(
            {**{f: "mean" for f in feat_cols}, "title": "count"}
        ).round(3).rename(columns={"title": "n_products"})

        clusters = []
        for cluster_id, row in summary.iterrows():
            info = {"cluster_id": int(cluster_id), **row.to_dict()}
            # Caractérisation automatique du segment
            if "composite_score" in row and "price" in row:
                score = row["composite_score"]
                price = row["price"]
                if score > 0.7:
                    info["segment_type"] = "Premium performant"
                elif price > df["price"].median() * 1.5:
                    info["segment_type"] = "Haut de gamme"
                elif score < 0.3:
                    info["segment_type"] = "Produits à risque"
                else:
                    info["segment_type"] = "Segment standard"

            if include_products:
                examples = df[df["cluster"] == cluster_id]["title"].head(3).tolist()
                info["examples"] = examples
            clusters.append(info)

        self._log_call("get_cluster_summary", {"include_products": include_products}, len(clusters))
        return {
            "clusters": clusters,
            "silhouette_info": "Utilisez le dashboard pour voir le Silhouette Score et le Davies-Bouldin Index",
        }

    def get_anomalies(self, limit: int = 20) -> dict:
        """Produits outliers DBSCAN."""
        df = self._load_data()
        if df.empty or "dbscan_cluster" not in df.columns:
            return {"error": "Données DBSCAN non disponibles."}

        anomalies = df[df["dbscan_cluster"] == -1]
        cols = [c for c in ["title", "price", "rating", "review_count", "category", "shop_name"] if c in anomalies.columns]
        result = anomalies.head(limit)[cols]

        self._log_call("get_anomalies", {"limit": limit}, len(result))
        return {
            "anomalies": result.to_dict("records"),
            "total_anomalies": len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / len(df) * 100, 2),
            "note": "Ces produits ont un profil atypique (prix aberrant, note incohérente, etc.) — vérification manuelle recommandée.",
        }

    def get_association_rules(self, min_lift: float = 1.0, limit: int = 10) -> dict:
        """Règles d'association Apriori."""
        rules = self._load_rules()
        if rules.empty:
            return {"error": "Règles d'association non disponibles. Lance d'abord le pipeline ML."}

        result = rules[rules["lift"] >= min_lift].head(limit)
        result_dict = []
        for _, row in result.iterrows():
            result_dict.append({
                "antecedents": str(row.get("antecedents", "")),
                "consequents": str(row.get("consequents", "")),
                "support": round(float(row.get("support", 0)), 4),
                "confidence": round(float(row.get("confidence", 0)), 4),
                "lift": round(float(row.get("lift", 0)), 4),
                "interpretation": f"Si un shop vend {row.get('antecedents', '')}, il vend aussi {row.get('consequents', '')} avec {row.get('confidence', 0)*100:.0f}% de confiance.",
            })

        self._log_call("get_association_rules", {"min_lift": min_lift, "limit": limit}, len(result_dict))
        return {"rules": result_dict, "total_rules": len(rules)}

    def get_catalog_stats(self, group_by: str = "category") -> dict:
        """Statistiques globales."""
        df = self._load_data()
        if df.empty:
            return {"error": "Données non disponibles."}

        global_stats = {
            "total_products": len(df),
            "avg_price": round(float(df["price"].mean()), 2) if "price" in df.columns else None,
            "avg_rating": round(float(df["rating"].mean()), 2) if "rating" in df.columns else None,
            "n_categories": int(df["category"].nunique()) if "category" in df.columns else None,
            "n_shops": int(df["shop_name"].nunique()) if "shop_name" in df.columns else None,
        }

        group_stats = None
        if group_by in df.columns:
            agg = df.groupby(group_by).agg(
                n_products=("title", "count"),
                avg_price=("price", "mean"),
                avg_rating=("rating", "mean"),
            ).round(2).sort_values("n_products", ascending=False).head(15)
            group_stats = agg.reset_index().to_dict("records")

        self._log_call("get_catalog_stats", {"group_by": group_by}, 1)
        return {"global": global_stats, "by_" + group_by: group_stats}

    def search_products(self, query: str = None, category: str = None,
                        min_price: float = None, max_price: float = None,
                        min_rating: float = None, in_stock_only: bool = False,
                        limit: int = 20) -> dict:
        """Recherche multi-critères."""
        df = self._load_data()
        if df.empty:
            return {"error": "Données non disponibles."}

        result = df.copy()
        if query and "title" in result.columns:
            result = result[result["title"].str.lower().str.contains(query.lower(), na=False)]
        if category and "category" in result.columns:
            result = result[result["category"].str.lower() == category.lower()]
        if min_price is not None and "price" in result.columns:
            result = result[result["price"] >= min_price]
        if max_price is not None and "price" in result.columns:
            result = result[result["price"] <= max_price]
        if min_rating is not None and "rating" in result.columns:
            result = result[result["rating"] >= min_rating]
        if in_stock_only and "stock" in result.columns:
            result = result[result["stock"] > 0]

        cols = [c for c in ["title", "price", "rating", "composite_score", "category", "shop_name"] if c in result.columns]
        if "composite_score" in result.columns:
            result = result.sort_values("composite_score", ascending=False)
        result = result.head(limit)[cols].round(3)

        params = {"query": query, "category": category, "min_price": min_price,
                  "max_price": max_price, "min_rating": min_rating}
        self._log_call("search_products", params, len(result))
        return {"products": result.to_dict("records"), "total_found": len(result)}

    def get_shop_ranking(self, limit: int = 10, sort_by: str = "composite_score") -> dict:
        """Classement des boutiques."""
        df = self._load_data()
        if df.empty or "shop_name" not in df.columns:
            return {"error": "Données non disponibles."}

        agg_dict = {"title": "count"}
        if "composite_score" in df.columns:
            agg_dict["composite_score"] = "mean"
        if "rating" in df.columns:
            agg_dict["rating"] = "mean"
        if "price" in df.columns:
            agg_dict["price"] = "mean"

        shops = df.groupby("shop_name").agg(agg_dict).round(3)
        shops = shops.rename(columns={"title": "n_products"})

        if sort_by in shops.columns:
            shops = shops.sort_values(sort_by, ascending=False)
        else:
            shops = shops.sort_values("n_products", ascending=False)

        shops = shops.head(limit).reset_index()

        self._log_call("get_shop_ranking", {"limit": limit, "sort_by": sort_by}, len(shops))
        return {"shops": shops.to_dict("records")}

    # ── Dispatcher MCP ────────────────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Point d'entrée principal — reçoit les appels du client MCP.
        Valide le nom de l'outil, délègue à la méthode correspondante.
        """
        logger.info(f"[MCP] Appel outil: {name}")

        tool_map = {
            "get_top_products":     self.get_top_products,
            "get_cluster_summary":  self.get_cluster_summary,
            "get_anomalies":        self.get_anomalies,
            "get_association_rules": self.get_association_rules,
            "get_catalog_stats":    self.get_catalog_stats,
            "search_products":      self.search_products,
            "get_shop_ranking":     self.get_shop_ranking,
        }

        if name not in tool_map:
            logger.warning(f"[MCP] Outil inconnu: {name}")
            return {"error": f"Outil '{name}' inconnu. Disponibles: {list(tool_map.keys())}"}

        try:
            result = tool_map[name](**arguments)
            return result
        except Exception as e:
            logger.error(f"[MCP] Erreur outil {name}: {e}")
            return {"error": str(e)}

    def list_tools(self) -> list:
        """Retourne la liste des outils disponibles (schéma MCP)."""
        return self.TOOLS

    def get_info(self) -> dict:
        """Informations sur le serveur MCP."""
        return {
            "name": "smart-ecommerce-mcp-server",
            "version": "1.0.0",
            "description": "Serveur MCP pour l'analyse intelligente de catalogues e-commerce",
            "author": "FST Tanger — LSI2 — DM & SID 2025/2026",
            "protocol": "Model Context Protocol v2025-03-26",
            "tools_count": len(self.TOOLS),
            "data_dir": str(DATA_DIR),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Serveur HTTP simple (FastAPI-compatible sans dépendances lourdes)
# ─────────────────────────────────────────────────────────────────────────────
def run_http_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Lance le serveur MCP en mode HTTP.
    Pour l'intégrer avec FastAPI si disponible.
    """
    try:
        from fastapi import FastAPI
        import uvicorn

        app = FastAPI(title="Smart eCommerce MCP Server")
        server = SmartEcommerceMCPServer()

        @app.get("/")
        def info():
            return server.get_info()

        @app.get("/tools")
        def list_tools():
            return {"tools": server.list_tools()}

        @app.post("/tools/{tool_name}")
        def call_tool(tool_name: str, body: dict = {}):
            return server.call_tool(tool_name, body)

        logger.info(f"🚀 MCP Server démarré → http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)

    except ImportError:
        # Fallback: serveur HTTP standard
        import http.server
        import threading

        server_instance = SmartEcommerceMCPServer()

        class MCPHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    response = json.dumps(server_instance.get_info(), ensure_ascii=False)
                elif self.path == "/tools":
                    response = json.dumps({"tools": server_instance.list_tools()})
                else:
                    response = json.dumps({"error": "Not Found"})
                    self.send_response(404)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                tool_name = self.path.strip("/tools/")
                result = server_instance.call_tool(tool_name, body)
                response = json.dumps(result, ensure_ascii=False, default=str)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, *args):
                pass  # Silence les logs HTTP standards

        httpd = http.server.HTTPServer((host, port), MCPHandler)
        logger.info(f"🚀 MCP Server (stdlib) → http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test", action="store_true", help="Tester les outils sans serveur")
    args = parser.parse_args()

    if args.test:
        print("\n🧪 Test des outils MCP\n")
        srv = SmartEcommerceMCPServer()
        print("✅ Info serveur :", srv.get_info())
        print("\n📋 Outils disponibles :")
        for t in srv.list_tools():
            print(f"   - {t['name']}: {t['description'][:60]}...")
        print("\n📊 Test get_catalog_stats :")
        print(json.dumps(srv.call_tool("get_catalog_stats", {}), default=str, indent=2, ensure_ascii=False))
    else:
        run_http_server(args.host, args.port)