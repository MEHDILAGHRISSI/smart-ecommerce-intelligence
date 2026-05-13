"""Configuration globale — chargée depuis le fichier .env."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemins utiles (accessibles directement en import)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"


class Settings(BaseSettings):
    """Paramètres du projet Smart eCommerce Intelligence."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Permet d'ignorer les variables .env non déclarées ici
    )

    # --- Scraping: Shopify ---
    SHOPIFY_BASE_URL: str = "https://hydrogen-preview.myshopify.com"
    SHOPIFY_API_KEY: Optional[str] = None

    # --- Scraping: WooCommerce ---
    WOO_BASE_URL: str = "https://demo.woothemes.com"
    WOO_CONSUMER_KEY: str = "ck_demo"
    WOO_CONSUMER_SECRET: str = "cs_demo"

    # --- Configuration Scraping ---
    MAX_SCRAPING_PAGES: int = 10
    REQUEST_TIMEOUT: float = 30.0
    REQUEST_DELAY: float = 0.5

    # --- Machine Learning & Data Mining ---
    TOP_K: int = 20
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    KMEANS_N_CLUSTERS: int = 5

    # --- Règles d'association ---
    MIN_SUPPORT: float = 0.1
    MIN_CONFIDENCE: float = 0.6

    # --- Intelligence Artificielle (LLM & MCP) ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEFAULT_LLM_MODEL: str = "gpt-4"  # ou "claude-3-opus-20240229"
    MCP_SERVER_PORT: int = 8000

    # --- Dashboard BI ---
    DASHBOARD_PORT: int = 8501


# =====================================================================
# Constantes directement importables (pour compatibilité pipeline)
# =====================================================================
_s = Settings()

# Rendre les dossiers accessibles et s'assurer qu'ils existent
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dépaquetage des variables ML & Data Mining pour un import direct
TOP_K             = _s.TOP_K
TEST_SIZE         = _s.TEST_SIZE
RANDOM_STATE      = _s.RANDOM_STATE
KMEANS_N_CLUSTERS = _s.KMEANS_N_CLUSTERS
MIN_SUPPORT       = _s.MIN_SUPPORT
MIN_CONFIDENCE    = _s.MIN_CONFIDENCE