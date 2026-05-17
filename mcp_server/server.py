"""
Serveur MCP officiel (Anthropic) pour Smart eCommerce Intelligence.

Ce module remplace l'ancien serveur HTTP artisanal. Il utilise le SDK MCP
Python avec les décorateurs :
- @server.list_tools()
- @server.call_tool()

Transport supporté:
- stdio (recommandé)
- sse (placeholder explicite ; stdio est opérationnel)
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ── Logging ──────────────────────────────────────────────────────────────────
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

DATA_DIR = Path("data/processed")


# ──────────────────────────────────────────────────────────────────────────────
# Validation stricte des arguments outils (Pydantic)
# ──────────────────────────────────────────────────────────────────────────────
class TopProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    limit: int = Field(default=10, ge=1, le=50)
    category: str | None = None
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ClusterSummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_products: bool = False


class AnomaliesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    limit: int = Field(default=20, ge=1, le=100)


class AssociationRulesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    min_lift: float = Field(default=1.0, ge=0.0)
    limit: int = Field(default=10, ge=1, le=100)


class CatalogStatsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    group_by: str = Field(default="category", pattern="^(category|shop_name|source_platform)$")


class SearchProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str | None = None
    category: str | None = None
    min_price: float | None = Field(default=None, ge=0.0)
    max_price: float | None = Field(default=None, ge=0.0)
    min_rating: float | None = Field(default=None, ge=0.0, le=5.0)
    in_stock_only: bool = False
    limit: int = Field(default=20, ge=1, le=200)


class ShopRankingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    limit: int = Field(default=10, ge=1, le=100)
    sort_by: str = Field(default="composite_score", pattern="^(composite_score|rating|n_products)$")


class CatalogAnalytics:
    """Service analytique mutualisé pour les outils MCP."""

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._rules: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        if self._df is None:
            path = DATA_DIR / "products_final.csv"
            if path.exists():
                self._df = pd.read_csv(path, encoding="utf-8")
                logger.info("Données chargées: %s produits", len(self._df))
            else:
                self._df = pd.DataFrame()
                logger.warning("products_final.csv introuvable")
        return self._df

    def _load_rules(self) -> pd.DataFrame:
        if self._rules is None:
            path = DATA_DIR / "association_rules.csv"
            self._rules = pd.read_csv(path, encoding="utf-8") if path.exists() else pd.DataFrame()
        return self._rules

    @staticmethod
    def _ok(data: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, **data}

    @staticmethod
    def _err(message: str) -> dict[str, Any]:
        return {"ok": False, "error": message}

    def get_top_products(self, args: TopProductsArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty:
            return self._err("Données indisponibles : exécuter le pipeline ML")

        result = df.copy()
        if "is_top_product" in result.columns:
            result = result[result["is_top_product"] == 1]
        if args.category and "category" in result.columns:
            result = result[result["category"].str.lower() == args.category.lower()]
        if args.min_score is not None and "composite_score" in result.columns:
            result = result[result["composite_score"] >= args.min_score]
        if "composite_score" in result.columns:
            result = result.sort_values("composite_score", ascending=False)

        cols = [
            c for c in [
                "title", "price", "rating", "review_count", "composite_score",
                "cluster_label", "price_anomaly_flag", "category", "shop_name",
            ] if c in result.columns
        ]
        out = result.head(args.limit)[cols].round(4)
        return self._ok({"total": len(out), "products": out.to_dict("records")})

    def get_cluster_summary(self, args: ClusterSummaryArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty or "cluster" not in df.columns:
            return self._err("Colonnes de clustering indisponibles")

        feat_cols = [c for c in ["price", "rating", "composite_score", "discount_score"] if c in df.columns]
        summary = (
            df.groupby(["cluster", "cluster_label"], dropna=False)[feat_cols + ["title"]]
            .agg({**{c: "mean" for c in feat_cols}, "title": "count"})
            .rename(columns={"title": "n_products"})
            .round(3)
            .reset_index()
        )
        clusters = summary.to_dict("records")

        if args.include_products:
            for cluster in clusters:
                cid = cluster["cluster"]
                sample = df[df["cluster"] == cid]["title"].head(3).tolist()
                cluster["examples"] = sample

        return self._ok({"clusters": clusters})

    def get_anomalies(self, args: AnomaliesArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty:
            return self._err("Données indisponibles")

        if "price_anomaly_flag" in df.columns:
            anomalies = df[df["price_anomaly_flag"] == 1]
        elif "dbscan_cluster" in df.columns:
            anomalies = df[df["dbscan_cluster"] == -1]
        else:
            return self._err("Aucune colonne d'anomalie détectée")

        cols = [c for c in ["title", "price", "rating", "price_anomaly_score", "category", "shop_name"] if c in anomalies.columns]
        out = anomalies.head(args.limit)[cols].round(4)
        return self._ok({
            "total_anomalies": int(len(anomalies)),
            "anomaly_rate_pct": round((len(anomalies) / len(df)) * 100, 2) if len(df) else 0.0,
            "anomalies": out.to_dict("records"),
        })

    def get_association_rules(self, args: AssociationRulesArgs) -> dict[str, Any]:
        rules = self._load_rules()
        if rules.empty:
            return self._err("Règles d'association indisponibles")

        result = rules[rules["lift"] >= args.min_lift].head(args.limit)
        payload = []
        for _, row in result.iterrows():
            payload.append({
                "antecedents": str(row.get("antecedents", "")),
                "consequents": str(row.get("consequents", "")),
                "support": round(float(row.get("support", 0.0)), 4),
                "confidence": round(float(row.get("confidence", 0.0)), 4),
                "lift": round(float(row.get("lift", 0.0)), 4),
            })
        return self._ok({"total": len(payload), "rules": payload})

    def get_catalog_stats(self, args: CatalogStatsArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty:
            return self._err("Données indisponibles")

        global_stats = {
            "total_products": len(df),
            "avg_price": round(float(df["price"].mean()), 2) if "price" in df.columns else None,
            "avg_rating": round(float(df["rating"].mean()), 2) if "rating" in df.columns else None,
            "n_categories": int(df["category"].nunique()) if "category" in df.columns else None,
            "n_shops": int(df["shop_name"].nunique()) if "shop_name" in df.columns else None,
        }

        by_group = []
        if args.group_by in df.columns:
            by_group = (
                df.groupby(args.group_by)
                .agg(n_products=("title", "count"), avg_price=("price", "mean"), avg_rating=("rating", "mean"))
                .round(3)
                .reset_index()
                .head(20)
                .to_dict("records")
            )
        return self._ok({"global": global_stats, f"by_{args.group_by}": by_group})

    def search_products(self, args: SearchProductsArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty:
            return self._err("Données indisponibles")

        result = df.copy()
        if args.query and "title" in result.columns:
            result = result[result["title"].str.contains(args.query, case=False, na=False)]
        if args.category and "category" in result.columns:
            result = result[result["category"].str.lower() == args.category.lower()]
        if args.min_price is not None and "price" in result.columns:
            result = result[result["price"] >= args.min_price]
        if args.max_price is not None and "price" in result.columns:
            result = result[result["price"] <= args.max_price]
        if args.min_rating is not None and "rating" in result.columns:
            result = result[result["rating"] >= args.min_rating]
        if args.in_stock_only:
            if "stock" in result.columns:
                result = result[result["stock"] > 0]
            elif "is_in_stock" in result.columns:
                result = result[result["is_in_stock"] == True]

        if "composite_score" in result.columns:
            result = result.sort_values("composite_score", ascending=False)

        cols = [c for c in ["title", "price", "rating", "composite_score", "cluster_label", "shop_name"] if c in result.columns]
        out = result.head(args.limit)[cols].round(4)
        return self._ok({"total_found": len(out), "products": out.to_dict("records")})

    def get_shop_ranking(self, args: ShopRankingArgs) -> dict[str, Any]:
        df = self._load_data()
        if df.empty or "shop_name" not in df.columns:
            return self._err("Données boutiques indisponibles")

        agg = {"title": "count"}
        for candidate in ["composite_score", "rating", "price"]:
            if candidate in df.columns:
                agg[candidate] = "mean"
        shops = df.groupby("shop_name").agg(agg).rename(columns={"title": "n_products"}).round(3)
        sort_col = args.sort_by if args.sort_by in shops.columns else "n_products"
        shops = shops.sort_values(sort_col, ascending=False).head(args.limit).reset_index()
        return self._ok({"shops": shops.to_dict("records")})


# ──────────────────────────────────────────────────────────────────────────────
# Handlers asynchrones pour les outils MCP
# ──────────────────────────────────────────────────────────────────────────────
async def handle_top_products(analytics: CatalogAnalytics, args: TopProductsArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_top_products."""
    def _sync_get_top_products():
        return analytics.get_top_products(args)
    return await asyncio.to_thread(_sync_get_top_products)


async def handle_cluster_summary(analytics: CatalogAnalytics, args: ClusterSummaryArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_cluster_summary."""
    def _sync_get_cluster_summary():
        return analytics.get_cluster_summary(args)
    return await asyncio.to_thread(_sync_get_cluster_summary)


async def handle_anomalies(analytics: CatalogAnalytics, args: AnomaliesArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_anomalies."""
    def _sync_get_anomalies():
        return analytics.get_anomalies(args)
    return await asyncio.to_thread(_sync_get_anomalies)


async def handle_association_rules(analytics: CatalogAnalytics, args: AssociationRulesArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_association_rules."""
    def _sync_get_association_rules():
        return analytics.get_association_rules(args)
    return await asyncio.to_thread(_sync_get_association_rules)


async def handle_catalog_stats(analytics: CatalogAnalytics, args: CatalogStatsArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_catalog_stats."""
    def _sync_get_catalog_stats():
        return analytics.get_catalog_stats(args)
    return await asyncio.to_thread(_sync_get_catalog_stats)


async def handle_search_products(analytics: CatalogAnalytics, args: SearchProductsArgs) -> dict[str, Any]:
    """Handler asynchrone pour search_products."""
    def _sync_search_products():
        return analytics.search_products(args)
    return await asyncio.to_thread(_sync_search_products)


async def handle_shop_ranking(analytics: CatalogAnalytics, args: ShopRankingArgs) -> dict[str, Any]:
    """Handler asynchrone pour get_shop_ranking."""
    def _sync_get_shop_ranking():
        return analytics.get_shop_ranking(args)
    return await asyncio.to_thread(_sync_get_shop_ranking)


def _tool_specs() -> list[types.Tool]:
    """Définit les métadonnées MCP des outils exposés au LLM."""
    return [
        types.Tool(name="get_top_products", description="Top produits par score ML", inputSchema=TopProductsArgs.model_json_schema()),
        types.Tool(name="get_cluster_summary", description="Résumé des segments KMeans", inputSchema=ClusterSummaryArgs.model_json_schema()),
        types.Tool(name="get_anomalies", description="Anomalies de prix (Isolation Forest / DBSCAN)", inputSchema=AnomaliesArgs.model_json_schema()),
        types.Tool(name="get_association_rules", description="Règles Apriori", inputSchema=AssociationRulesArgs.model_json_schema()),
        types.Tool(name="get_catalog_stats", description="Statistiques globales catalogue", inputSchema=CatalogStatsArgs.model_json_schema()),
        types.Tool(name="search_products", description="Recherche produits multi-critères", inputSchema=SearchProductsArgs.model_json_schema()),
        types.Tool(name="get_shop_ranking", description="Classement des boutiques", inputSchema=ShopRankingArgs.model_json_schema()),
    ]


class SmartEcommerceMCPServer:
    """Wrapper local synchrone pour le dashboard et les appels internes."""

    def __init__(self) -> None:
        self.analytics = CatalogAnalytics()
        self._validators: dict[str, tuple[type[BaseModel], Callable[[BaseModel], dict[str, Any]]]] = {
            "get_top_products": (TopProductsArgs, self.analytics.get_top_products),
            "get_cluster_summary": (ClusterSummaryArgs, self.analytics.get_cluster_summary),
            "get_anomalies": (AnomaliesArgs, self.analytics.get_anomalies),
            "get_association_rules": (AssociationRulesArgs, self.analytics.get_association_rules),
            "get_catalog_stats": (CatalogStatsArgs, self.analytics.get_catalog_stats),
            "search_products": (SearchProductsArgs, self.analytics.search_products),
            "get_shop_ranking": (ShopRankingArgs, self.analytics.get_shop_ranking),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Retourne les schémas d'outils au format Pydantic/JSON Schema."""
        return [
            {
                "name": name,
                "description": description,
                "inputSchema": model_cls.model_json_schema(),
            }
            for name, (model_cls, _handler) in self._validators.items()
            for description in [
                {
                    "get_top_products": "Top produits par score ML",
                    "get_cluster_summary": "Résumé des segments KMeans",
                    "get_anomalies": "Anomalies de prix (Isolation Forest / DBSCAN)",
                    "get_association_rules": "Règles Apriori",
                    "get_catalog_stats": "Statistiques globales catalogue",
                    "search_products": "Recherche produits multi-critères",
                    "get_shop_ranking": "Classement des boutiques",
                }[name]
            ]
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Exécute un outil localement avec validation Pydantic stricte."""
        if name not in self._validators:
            return {"ok": False, "error": f"Outil inconnu: {name}"}

        model_cls, handler = self._validators[name]
        try:
            args_obj = model_cls.model_validate(arguments or {})
            logger.info("TOOL=%s ARGS=%s", name, args_obj.model_dump())
            return handler(args_obj)
        except ValidationError as exc:
            return {"ok": False, "error": "Paramètres invalides", "details": exc.errors()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur outil MCP %s", name)
            return {"ok": False, "error": str(exc)}


def build_server() -> tuple[Server, list[types.Tool]]:
    """Construit le serveur MCP et configure ses handlers asynchrones."""
    server = Server("smart-ecommerce-mcp")
    analytics = CatalogAnalytics()
    tool_specs = _tool_specs()

    # Mappage des outils aux handlers asynchrones
    validators: dict[str, tuple[type[BaseModel], Callable[[BaseModel], Any]]] = {
        "get_top_products": (TopProductsArgs, lambda args: handle_top_products(analytics, args)),
        "get_cluster_summary": (ClusterSummaryArgs, lambda args: handle_cluster_summary(analytics, args)),
        "get_anomalies": (AnomaliesArgs, lambda args: handle_anomalies(analytics, args)),
        "get_association_rules": (AssociationRulesArgs, lambda args: handle_association_rules(analytics, args)),
        "get_catalog_stats": (CatalogStatsArgs, lambda args: handle_catalog_stats(analytics, args)),
        "search_products": (SearchProductsArgs, lambda args: handle_search_products(analytics, args)),
        "get_shop_ranking": (ShopRankingArgs, lambda args: handle_shop_ranking(analytics, args)),
    }

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """Retourne la liste des outils disponibles."""
        return tool_specs

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Exécute un outil de manière complètement asynchrone.

        Les handlers sont des coroutines qui exécutent les opérations bloquantes
        (fichier I/O) via asyncio.to_thread() pour préserver la réactivité du
        protocole stdio.
        """
        if name not in validators:
            return {"ok": False, "error": f"Outil inconnu: {name}"}

        model_cls, handler_factory = validators[name]
        try:
            args_obj = model_cls.model_validate(arguments or {})
            logger.info("TOOL=%s ARGS=%s", name, args_obj.model_dump())

            # Créer le handler (peut être une couroutine)
            handler_result = handler_factory(args_obj)

            # Vérifier si le handler est une coroutine et l'attendre
            if inspect.iscoroutine(handler_result):
                result = await handler_result
            else:
                result = handler_result

            return result
        except ValidationError as exc:
            return {"ok": False, "error": "Paramètres invalides", "details": exc.errors()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erreur outil MCP %s", name)
            return {"ok": False, "error": str(exc)}

    return server, tool_specs


async def run_stdio() -> None:
    """Exécute le serveur MCP sur stdio (transport recommandé).

    Utilise l'SDK MCP officiel avec transport stdio pour la communication
    avec les clients MCP (Claude Desktop, Cursor, etc.).
    """
    server, _ = build_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart eCommerce MCP Server (Official SDK)")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    args = parser.parse_args()

    if args.transport == "sse":
        raise SystemExit(
            "Transport SSE non activé dans cette version. Utilise --transport stdio "
            "(conforme MCP et recommandé pour les hosts locaux)."
        )


    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()

