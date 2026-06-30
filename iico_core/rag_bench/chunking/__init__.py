"""
iico_core/rag_bench/chunking/__init__.py
=========================================
Plugins de chunking intercambiables.
"""
from .base import (
    ChunkBoundary,
    ChunkingResult,
    ChunkingStrategy,
    register_chunker,
    get_chunker,
    list_chunkers,
)
from .pipeline import ChunkingPipeline

# Importar para registrar en el registry automáticamente
from . import document     # noqa: F401
from . import naive        # noqa: F401
from . import structural   # noqa: F401
from . import semantic     # noqa: F401
from . import convolution  # noqa: F401

__all__ = [
    "ChunkBoundary",
    "ChunkingResult",
    "ChunkingStrategy",
    "ChunkingPipeline",
    "register_chunker",
    "get_chunker",
    "list_chunkers",
]
