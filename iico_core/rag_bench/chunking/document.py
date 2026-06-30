from __future__ import annotations

from typing import Any

from .base import ChunkBoundary, ChunkingResult, ChunkingStrategy, register_chunker

@register_chunker
class DocumentChunker(ChunkingStrategy):
    """Chunker que no divide el texto.
    
    Devuelve la nota completa como un único chunk. Se utiliza como línea base
    (baseline) para demostrar el peor rendimiento posible cuando el LLM
    recibe demasiado contexto innecesario.
    """
    name = "document"
    description = "Devuelve el documento completo sin aplicar segmentación."

    def chunk(self, text: str, config: dict[str, Any] | None = None) -> ChunkingResult:
        # Un solo chunk con el texto completo y sin boundaries
        return ChunkingResult(
            fragments=[text.strip()],
            boundaries=[]
        )
