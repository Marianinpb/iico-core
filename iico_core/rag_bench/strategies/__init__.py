"""
iico_core/rag_bench/strategies/__init__.py
===========================================
Plugins de retrieval intercambiables.
"""
from .base import (
    RetrievalStrategy,
    register_strategy,
    get_strategy,
    list_strategies,
)

# Registrar automáticamente al importar el paquete
from . import embedding_strategy  # noqa: F401
from . import splay_strategy       # noqa: F401

__all__ = [
    "RetrievalStrategy",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
