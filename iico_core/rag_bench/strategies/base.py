"""
iico_core/rag_bench/strategies/base.py
========================================
Clase base abstracta para estrategias de retrieval + registry automático.

Agregar una nueva estrategia::

    from iico_core.rag_bench.strategies.base import RetrievalStrategy, register_strategy

    @register_strategy
    class MiEstrategia(RetrievalStrategy):
        name = "mi_estrategia"
        description = "Descripción corta para reportes"

        def setup(self, db, embedding_index, config):
            ...

        def retrieve(self, query, top_k=5):
            ...
            return [(chunk_dict, score), ...]

Aparece automáticamente en el CLI y el runner sin cambiar ningún otro archivo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrievalStrategy(ABC):
    """Interfaz base para todas las estrategias de retrieval.

    Contrato::

        setup()    → inicializar con BD + EmbeddingIndex
        retrieve() → [(chunk_dict, score), ...]  ordenado por score desc
        teardown() → limpieza opcional
    """

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def setup(
        self,
        db: Any,                   # NoteDB
        embedding_index: Any,      # EmbeddingIndex (puede ser None)
        config: dict,
    ) -> None:
        """Inicializa la estrategia con la BD y el índice.

        Args:
            db: instancia de :class:`NoteDB`.
            embedding_index: instancia de :class:`EmbeddingIndex`, o ``None``
                             si la estrategia no requiere embeddings.
            config: configuración específica de la estrategia.
        """
        ...

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """Recupera chunks relevantes para el query.

        Args:
            query: texto de la consulta.
            top_k: número máximo de resultados.

        Returns:
            Lista de ``(chunk_dict, score)`` ordenada por score descendente.
            ``chunk_dict`` tiene las mismas keys que ``NoteDB.get_all_chunks()``.
            ``score`` está en ``[0, 1]`` (mayor = más relevante).
        """
        ...

    def teardown(self) -> None:
        """Limpieza opcional de recursos."""
        pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type[RetrievalStrategy]] = {}


def register_strategy(cls: type[RetrievalStrategy]) -> type[RetrievalStrategy]:
    """Decorador que registra una estrategia en el registry global."""
    if not hasattr(cls, "name") or cls.name == "unnamed":
        raise ValueError(
            f"La estrategia {cls.__name__} debe definir un atributo 'name'."
        )
    _STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[RetrievalStrategy]:
    """Devuelve la clase de estrategia por nombre.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if name not in _STRATEGY_REGISTRY:
        available = list_strategies()
        raise KeyError(
            f"Estrategia '{name}' no encontrada. "
            f"Disponibles: {available}"
        )
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Devuelve los nombres de todas las estrategias registradas."""
    return list(_STRATEGY_REGISTRY.keys())
