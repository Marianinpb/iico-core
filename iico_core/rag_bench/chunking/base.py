"""
iico_core/rag_bench/chunking/base.py
======================================
Clase base abstracta para estrategias de chunking + registry automático.

Agregar un nuevo chunker es tan simple como::

    from iico_core.rag_bench.chunking.base import ChunkingStrategy, register_chunker

    @register_chunker
    class MiChunker(ChunkingStrategy):
        name = "mi_chunker"
        description = "Descripción corta"

        def chunk(self, text, config=None):
            ...

El decorador ``@register_chunker`` lo hace aparecer automáticamente
en el pipeline y en el CLI ``--chunking mi_chunker``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class ChunkBoundary:
    """Un punto de corte detectado por un chunker.

    Attributes:
        position: Índice de la unidad (oración/párrafo) *después* de la cual
                  se hace el corte. Ejemplo: position=2 → cortar después de
                  la unidad 2.
        confidence: Qué tan seguro está el chunker del corte (0.0 – 1.0).
                    Un valle muy profundo → alta confianza.
        reason: Descripción legible del motivo del corte.
                Ejemplos: "header_h2", "kernel_valley(depth=0.21)".
    """
    position: int
    confidence: float
    reason: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ChunkingResult:
    """Resultado completo de una estrategia de chunking.

    Attributes:
        fragments: Lista de textos resultantes del corte.
        boundaries: Puntos de corte detectados (uno por separación).
        metadata: Información extra específica del chunker.
                  Los chunkers DSP exportan aquí sus señales intermedias
                  para que el ReportGenerator pueda graficarlas.

    Ejemplo de ``metadata`` para ConvolutionChunker::

        {
            "chunker": "convolution",
            "kernel_type": "gaussian",
            "kernel_size": 5,
            "similarity_signal": [0.82, 0.79, 0.31, 0.88, ...],
            "smoothed_signal":   [0.80, 0.73, 0.40, 0.75, ...],
            "valley_threshold":  0.5,
            "valley_positions":  [2],
            "num_units": 6,
        }
    """
    fragments: list[str]
    boundaries: list[ChunkBoundary]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.fragments)

    @property
    def has_dsp_signals(self) -> bool:
        """True si el chunker exportó señales DSP."""
        return "similarity_signal" in self.metadata


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class ChunkingStrategy(ABC):
    """Interfaz base para todas las estrategias de chunking.

    Subclases deben:
    1. Definir atributos de clase ``name`` y ``description``.
    2. Decorar con ``@register_chunker``.
    3. Implementar ``chunk()``.
    4. Implementar opcionalmente ``setup()`` si necesitan el EmbeddingIndex.
    """

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def chunk(self, text: str, config: dict | None = None) -> ChunkingResult:
        """Divide un texto en fragmentos.

        Args:
            text: Texto completo a dividir (cuerpo Markdown de la nota).
            config: Parámetros de configuración del chunker.
                    Cada chunker define sus propias claves.

        Returns:
            :class:`ChunkingResult` con fragmentos, puntos de corte
            y metadata (incluyendo señales DSP si aplica).
        """
        ...

    def setup(self, embedding_index: Any = None, **kwargs: Any) -> None:
        """Inicialización opcional.

        Llamada por :class:`ChunkingPipeline` antes de procesar notas.
        Úsala para cargar el EmbeddingIndex o cualquier recurso pesado.
        """
        pass

    def teardown(self) -> None:
        """Limpieza opcional de recursos."""
        pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CHUNKER_REGISTRY: dict[str, type[ChunkingStrategy]] = {}


def register_chunker(cls: type[ChunkingStrategy]) -> type[ChunkingStrategy]:
    """Decorador que registra un chunker en el registry global.

    Example::

        @register_chunker
        class MiChunker(ChunkingStrategy):
            name = "mi_chunker"
    """
    if not hasattr(cls, "name") or cls.name == "unnamed":
        raise ValueError(
            f"El chunker {cls.__name__} debe definir un atributo 'name' "
            "distinto de 'unnamed'."
        )
    _CHUNKER_REGISTRY[cls.name] = cls
    return cls


def get_chunker(name: str) -> type[ChunkingStrategy]:
    """Devuelve la clase de chunker por nombre.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if name not in _CHUNKER_REGISTRY:
        available = list_chunkers()
        raise KeyError(
            f"Chunker '{name}' no encontrado. "
            f"Disponibles: {available}"
        )
    return _CHUNKER_REGISTRY[name]


def list_chunkers() -> list[str]:
    """Devuelve los nombres de todos los chunkers registrados."""
    return list(_CHUNKER_REGISTRY.keys())
