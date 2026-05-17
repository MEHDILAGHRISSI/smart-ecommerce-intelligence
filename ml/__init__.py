"""Package ML du projet Smart eCommerce Intelligence."""

from __future__ import annotations

# Export minimal pour garder les imports stables dans les tests et scripts.
from .clustering import cluster_products

__all__ = ["cluster_products"]

