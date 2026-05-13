"""Driver Playwright asynchrone réutilisable par tous les agents."""

from __future__ import annotations

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from loguru import logger


class PlaywrightDriver:
    """
    Gestionnaire de contexte async pour Playwright (Chromium headless).

    Utilisation :
        async with PlaywrightDriver() as driver:
            page = await driver.new_page()
            await page.goto("https://example.com")
            html = await page.content()
    """

    def __init__(self, headless: bool = True, timeout: int = 30_000) -> None:
        """
        Initialise le driver.

        Args:
            headless: Lancer le navigateur en mode invisible (défaut: True)
            timeout: Timeout en millisecondes pour les actions Playwright (défaut: 30s)
        """
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "PlaywrightDriver":
        """Lance Playwright et ouvre le navigateur Chromium."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            user_agent="SmartEcommerceBot/1.0 (University Project FST Tanger)",
            extra_http_headers={"Accept-Language": "fr-FR,fr;q=0.9"},
        )
        self._context.set_default_timeout(self.timeout)
        logger.debug(f"Navigateur Chromium lancé (headless={self.headless})")
        return self

    async def __aexit__(self, *args) -> None:
        """Ferme proprement le navigateur et Playwright."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.debug("Navigateur Chromium fermé proprement")

    async def new_page(self) -> Page:
        """
        Crée et retourne une nouvelle page dans le contexte actuel.

        Returns:
            Une instance Page Playwright prête à l'emploi

        Raises:
            RuntimeError: Si le driver n'est pas initialisé via context manager
        """
        if not self._context:
            raise RuntimeError("PlaywrightDriver non initialisé. Utilise 'async with'.")
        page = await self._context.new_page()
        logger.debug("Nouvelle page Playwright créée")
        return page

    async def fetch_page_source(self, url: str) -> str:
        """
        Navigue vers une URL et retourne le contenu HTML complet.

        Args:
            url: URL cible

        Returns:
            Contenu HTML de la page après chargement complet
        """
        page = await self.new_page()
        try:
            logger.info(f"Playwright → navigation vers {url}")
            await page.goto(url, wait_until="domcontentloaded")
            content = await page.content()
            return content
        finally:
            await page.close()