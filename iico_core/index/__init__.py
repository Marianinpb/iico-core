"""
iico_core/index/__init__.py
Expone SplayTree, SplayCacheMetrics y EmbeddingIndex como interfaz pública.
"""

from .splay_tree import SplayCacheMetrics, SplayTree

__all__ = ["SplayTree", "SplayCacheMetrics"]
