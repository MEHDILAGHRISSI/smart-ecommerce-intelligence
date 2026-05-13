"""Exceptions personnalisées pour les agents de scraping."""

class ScrapingError(Exception):
    """Exception de base pour les erreurs de scraping."""
    pass

class APIUnavailableError(ScrapingError):
    """Levée lorsque l'API de la plateforme est inaccessible ou retourne une erreur."""
    pass

class NormalizationError(ScrapingError):
    """Levée lorsque les données brutes ne peuvent pas être formatées selon le ProductSchema."""
    pass