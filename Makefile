# Smart eCommerce Intelligence — Makefile
# FST Tanger — LSI2 — DM & SID 2025/2026

.PHONY: help install run demo run-real dashboard test lint kubeflow mcp docker clean

# Couleurs
BLUE=\033[0;34m
GREEN=\033[0;32m
YELLOW=\033[0;33m
RED=\033[0;31m
NC=\033[0m

help:
	@echo ""
	@echo "$(BLUE)Smart eCommerce Intelligence$(NC)"
	@echo "FST Tanger — LSI2 — DM & SID 2025/2026"
	@echo ""
	@echo "$(GREEN)Commandes disponibles :$(NC)"
	@echo ""
	@echo "  $(YELLOW)make install$(NC)       Installer les dépendances"
	@echo "  $(YELLOW)make run$(NC)           Pipeline complet (2000 produits synthétiques)"
	@echo "  $(YELLOW)make run-real$(NC)      Pipeline avec scraping réel Shopify"
	@echo "  $(YELLOW)make demo$(NC)          Démo rapide (200 produits)"
	@echo "  $(YELLOW)make dashboard$(NC)     Lancer uniquement le dashboard"
	@echo "  $(YELLOW)make test$(NC)          Lancer les tests"
	@echo "  $(YELLOW)make lint$(NC)          Vérification qualité du code"
	@echo "  $(YELLOW)make docker$(NC)        Lancer via Docker Compose"
	@echo "  $(YELLOW)make clean$(NC)         Nettoyer les données"
	@echo ""

install:
	@echo "$(BLUE)📦 Installation des dépendances...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Installation terminée$(NC)"

# ── Pipeline principal ────────────────────────────────────────────────────────

run: _check-env
	@echo "$(BLUE)🚀 Démarrage du pipeline complet...$(NC)"
	@echo "$(YELLOW)Étape 1/3 : Génération de 2000 produits synthétiques$(NC)"
	PYTHONPATH=. python data/generate_synthetic.py 2000
	@echo "$(YELLOW)Étape 2/3 : Exécution du pipeline ML$(NC)"
	PYTHONPATH=. python ml/pipeline.py
	@echo "$(YELLOW)Étape 3/3 : Lancement du dashboard$(NC)"
	PYTHONPATH=. streamlit run dashboard/app.py
	@echo "$(GREEN)✅ Pipeline terminé avec succès !$(NC)"

run-real: _check-env
	@echo "$(BLUE)🚀 Pipeline avec données réelles (Shopify)...$(NC)"
	@echo "$(YELLOW)Étape 1/3 : Scraping réel$(NC)"
	PYTHONPATH=. python data/scrape_real_stores.py --limit 5 --pages 3
	@echo "$(YELLOW)Étape 2/3 : Pipeline ML$(NC)"
	PYTHONPATH=. python ml/pipeline.py
	@echo "$(YELLOW)Étape 3/3 : Dashboard$(NC)"
	PYTHONPATH=. streamlit run dashboard/app.py

demo: _check-env
	@echo "$(BLUE)⚡ Démo rapide (200 produits)...$(NC)"
	PYTHONPATH=. python data/generate_synthetic.py 200
	PYTHONPATH=. python ml/pipeline.py
	PYTHONPATH=. streamlit run dashboard/app.py

dashboard:
	@echo "$(BLUE)📊 Lancement du dashboard Streamlit...$(NC)"
	PYTHONPATH=. streamlit run dashboard/app.py --server.port=8501

# ── Tests & Qualité ───────────────────────────────────────────────────────────

test:
	@echo "$(BLUE)🧪 Lancement des tests unitaires...$(NC)"
	PYTHONPATH=. pytest tests/ -v --tb=short

test-coverage:
	@echo "$(BLUE)🧪 Tests avec couverture de code...$(NC)"
	PYTHONPATH=. pytest tests/ -v --cov=ml --cov=agents --cov=configs \
		--cov-report=term-missing --cov-report=html

lint:
	@echo "$(BLUE)🔍 Analyse qualité du code...$(NC)"
	ruff check . --fix
	ruff format .
	@echo "$(GREEN)✅ Code formaté et vérifié$(NC)"

# ── Docker ────────────────────────────────────────────────────────────────────

docker:
	@echo "$(BLUE)🐳 Lancement via Docker Compose...$(NC)"
	docker-compose up --build

docker-stop:
	docker-compose down

# ── Nettoyage ─────────────────────────────────────────────────────────────────

clean:
	@echo "$(YELLOW)🧹 Nettoyage des données générées...$(NC)"
	rm -rf data/raw/*.json
	rm -rf data/processed/*.csv
	rm -rf ml/models/*.joblib
	@echo "$(GREEN)✅ Données nettoyées$(NC)"

clean-all: clean
	@echo "$(YELLOW)🧹 Nettoyage complet...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

# ── Helpers ───────────────────────────────────────────────────────────────────

_check-env:
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)⚠️ Fichier .env manquant → création automatique...$(NC)"; \
		echo "GEMINI_API_KEY=ta_clé_gemini_ici" > .env; \
		echo "TOP_K=20" >> .env; \
		echo "KMEANS_N_CLUSTERS=5" >> .env; \
		echo "MIN_SUPPORT=0.1" >> .env; \
		echo "MIN_CONFIDENCE=0.6" >> .env; \
		echo "$(GREEN)✅ .env créé$(NC)"; \
	fi

status:
	@echo "$(BLUE)📊 État actuel du projet$(NC)"
	@echo "Données brutes     :" `ls data/raw/*.json 2>/dev/null | wc -l || echo 0` "fichiers"
	@echo "Données traitées   :" `ls data/processed/*.csv 2>/dev/null | wc -l || echo 0` "fichiers"
	@echo "Modèles ML         :" `ls ml/models/*.joblib 2>/dev/null | wc -l || echo 0` "fichiers"