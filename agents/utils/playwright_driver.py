"""
agents/utils/playwright_driver.py — Smart eCommerce Intelligence
================================================================
Driver Playwright asynchrone réutilisable par tous les agents.
CORRECTION v2 — Anti-détection renforcée (Cloudflare / Turnstile) :
  - playwright-stealth pour evasion complète (Canvas, WebGL, WebRTC, etc.)
  - User-Agent réaliste Chrome 120 Windows
  - En-têtes HTTP stealth
  - Chromium args + JS injection (double sécurité)
  - Viewport / Locale / Timezone réalistes
"""

from __future__ import annotations
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright_stealth import stealth_async   # ← Ajouté pour stealth complet
from loguru import logger

# User-Agent Chrome 120 réaliste — identique à un navigateur Windows ordinaire.
_REAL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# En-têtes HTTP réalistes
_STEALTH_HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-CH-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

# Arguments Chromium anti-détection
_CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
]


class PlaywrightDriver:
    """
    Gestionnaire de contexte async pour Playwright avec stealth renforcé.
    """

    def __init__(self, headless: bool = True, timeout: int = 30_000) -> None:
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "PlaywrightDriver":
        """Lance Playwright et ouvre Chromium avec profil anti-détection."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=_CHROMIUM_ARGS,
        )
        self._context = await self._browser.new_context(
            user_agent=_REAL_USER_AGENT,
            extra_http_headers=_STEALTH_HEADERS,
            viewport={"width": 1366, "height": 768},
            locale="fr-FR",
            timezone_id="Africa/Casablanca",
            java_script_enabled=True,
        )
        self._context.set_default_timeout(self.timeout)

        logger.debug(
            f"Chromium lancé (headless={self.headless}) | "
            f"UA: Chrome/120 Windows | Stealth: playwright-stealth activé"
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Ferme proprement le navigateur et Playwright."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.debug("Chromium fermé proprement")

    async def new_page(self) -> Page:
        """
        Crée une nouvelle page avec stealth complet.
        """
        if not self._context:
            raise RuntimeError(
                "PlaywrightDriver non initialisé. Utilisez 'async with PlaywrightDriver() as driver'."
            )

        page = await self._context.new_page()

        # === STEALTH COMPLET ===
        await stealth_async(page)                    # ← playwright-stealth principal

        # Double sécurité : injection JS explicite (comme demandé)
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        logger.debug("Nouvelle page Playwright créée avec stealth complet")
        return page

    async def fetch_page_source(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
    ) -> str:
        """
        Navigue vers une URL et retourne le contenu HTML complet.
        """
        page = await self.new_page()
        try:
            logger.info(f"Playwright → {url} (wait_until={wait_until})")
            await page.goto(url, wait_until=wait_until)
            content = await page.content()
            return content
        except Exception as exc:
            logger.warning(f"Playwright → Erreur sur {url} : {exc}")
            raise
        finally:
            await page.close()