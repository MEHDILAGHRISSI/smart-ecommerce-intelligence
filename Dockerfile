FROM python:3.11-slim

# Définir le dossier de travail
WORKDIR /app

# Éviter que Python n'écrive des fichiers .pyc et forcer l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer manuellement les dépendances système requises pour Chromium Headless (Playwright)
# Ajout de libpango-1.0-0 et libcairo2 demandés par l'avertissement Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement le fichier des dépendances dans un premier temps (optimisation du cache Docker)
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Installer les binaires de Chromium pour Playwright
RUN playwright install chromium

# Copier le reste du code du projet
COPY . .

# Déclarer la commande par défaut pour lancer votre serveur MCP
CMD ["python", "data/overnight_enrichment.py"]