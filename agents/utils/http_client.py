"""Client HTTP asynchrone partagé entre tous les agents."""

from __future__ import annotations

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _build_retry_decorator():
    return retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda rs: logger.warning(
            f"Tentative {rs.attempt_number} échouée, retry dans {rs.next_action.sleep:.1f}s..."
        ),
    )


network_retry = _build_retry_decorator()


class HttpClient:
    """
    Client httpx asynchrone réutilisable avec headers et timeout configurables.

    Utilisation via context manager async :
        async with HttpClient(base_url="https://...") as client:
            data = await client.get("/products")
    """

    DEFAULT_TIMEOUT = 30.0
    DEFAULT_HEADERS = {
        "User-Agent": "SmartEcommerceBot/1.0 (University Project FST Tanger)",
        "Accept": "application/json",
    }

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "HttpClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
        logger.debug(f"Session HTTP ouverte → {self._base_url}")
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            logger.debug(f"Session HTTP fermée → {self._base_url}")

    @network_retry
    async def get(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Effectue une requête GET avec retry automatique (3 tentatives, backoff exponentiel)."""
        if not self._client:
            raise RuntimeError("HttpClient non initialisé. Utilise 'async with'.")
        logger.debug(f"GET {self._base_url}{endpoint} | params={params}")
        response = await self._client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()