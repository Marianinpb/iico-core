from __future__ import annotations

from typing import Any

from .base import ChunkBoundary, ChunkingResult, ChunkingStrategy, register_chunker

@register_chunker
class NaiveChunker(ChunkingStrategy):
    """Chunker ingenuo por tamaño constante.
    
    Divide el texto estáticamente en fragmentos de tamaño fijo basados en el 
    límite de tokens, intentando cortar por espacios para no romper palabras.
    
    Config keys:
        max_chunk_tokens (int): Máximo de tokens por chunk. Default: 512.
    """
    name = "naive"
    description = "Corta el texto en pedazos de tamaño constante."

    def chunk(self, text: str, config: dict[str, Any] | None = None) -> ChunkingResult:
        config = config or {}
        # Asumimos ~4 caracteres por token
        max_tokens = config.get("max_chunk_tokens", 512)
        max_chars = max_tokens * 4

        words = text.split()
        fragments = []
        boundaries = []
        
        current_chunk = []
        current_len = 0
        
        for i, word in enumerate(words):
            word_len = len(word) + 1  # +1 por el espacio
            if current_len + word_len > max_chars and current_chunk:
                fragments.append(" ".join(current_chunk))
                # Un boundary aproximado apuntando a la posición de la palabra
                boundaries.append(ChunkBoundary(position=i, confidence=1.0, reason="token_limit"))
                current_chunk = [word]
                current_len = word_len
            else:
                current_chunk.append(word)
                current_len += word_len
                
        if current_chunk:
            fragments.append(" ".join(current_chunk))
            
        return ChunkingResult(
            fragments=fragments,
            boundaries=boundaries
        )
