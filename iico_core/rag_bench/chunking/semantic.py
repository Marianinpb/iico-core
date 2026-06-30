from __future__ import annotations

import re
from typing import Any

from .base import ChunkBoundary, ChunkingResult, ChunkingStrategy, register_chunker

@register_chunker
class SemanticChunker(ChunkingStrategy):
    """Chunker semántico basado en embeddings.
    
    Divide el texto en oraciones y calcula el vector (embedding) de cada una.
    Si la similitud del coseno entre dos oraciones consecutivas cae por debajo
    de un umbral, se crea un límite (boundary) de chunk, asumiendo un cambio
    de tema.
    
    Config keys:
        similarity_threshold (float): Umbral por debajo del cual se corta. Default: 0.5.
        min_sentences (int): Mínimo de oraciones por chunk. Default: 1.
    """
    name = "semantic"
    description = "Divide el texto dinámicamente detectando caídas de similitud semántica."

    def __init__(self) -> None:
        self._index = None

    def setup(self, embedding_index: Any = None, **kwargs: Any) -> None:
        self._index = embedding_index

    def chunk(self, text: str, config: dict[str, Any] | None = None) -> ChunkingResult:
        if not self._index:
            raise RuntimeError("SemanticChunker requiere un EmbeddingIndex inicializado.")
            
        config = config or {}
        threshold = config.get("similarity_threshold", 0.5)
        min_sentences = config.get("min_sentences", 1)

        # Dividir heurísticamente en oraciones (por puntos seguidos de espacios o saltos de línea)
        raw_sentences = re.split(r'(?<=[.?!])\s+|\n+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return ChunkingResult(fragments=[], boundaries=[])
        if len(sentences) == 1:
            return ChunkingResult(fragments=sentences, boundaries=[])

        # Import local para evitar dependencia fuerte si se llama sin index (ya falló arriba igual)
        from ...index.embedding import cosine_similarity

        # Vectorizar todas las oraciones
        embeddings = [self._index.vectorize(s) for s in sentences]

        fragments = []
        boundaries = []
        
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i-1], embeddings[i])
            
            # Cortar si la similitud es baja Y ya tenemos el mínimo de oraciones
            if sim < threshold and len(current_chunk) >= min_sentences:
                fragments.append(" ".join(current_chunk))
                boundaries.append(ChunkBoundary(position=i-1, confidence=1.0-sim, reason=f"semantic_drop({sim:.2f})"))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            fragments.append(" ".join(current_chunk))

        return ChunkingResult(
            fragments=fragments,
            boundaries=boundaries,
            metadata={
                "chunker": "semantic",
                "threshold": threshold,
            }
        )
