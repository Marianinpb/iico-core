"""
iico_core/memory/passive.py
============================
Sistema de Memoria Pasiva: notas Markdown con frontmatter YAML.

Cada nota es un archivo .md en memory_store/ con esta estructura:

    ---
    id: protocolo-spi
    tags: [spi, comunicación, hardware]
    priority: 5          # 1-10, mayor = más relevante
    created: 2026-06-04
    ---
    # Protocolo SPI
    El SPI (Serial Peripheral Interface) es un bus de comunicación síncrono...

El Harness carga todas las notas al iniciar y las inyecta al system prompt
según relevancia (tags en Fase 1, embeddings en Fase 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import frontmatter  # python-frontmatter

if TYPE_CHECKING:
    from ..types import Chunk

from ..types import Chunk
from .chunk_store import ChunkStore
from .chunker import Chunker


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------

@dataclass
class PassiveNote:
    """Representa una nota de la Memoria Pasiva."""
    id: str
    tags: list[str]
    priority: int          # 1-10
    content: str           # Cuerpo Markdown (sin el frontmatter)
    source_path: Path | None = None

    def token_estimate(self) -> int:
        """Estimación rápida de tokens (1 token ≈ 4 caracteres en español)."""
        return len(self.content) // 4


# ---------------------------------------------------------------------------
# Memoria Pasiva
# ---------------------------------------------------------------------------

class PassiveMemory:
    """
    Gestor de notas Markdown+YAML.
    
    Fase 1: búsqueda determinista por tags (O(n), sin dependencias externas).
    Fase 2: se agregará búsqueda semántica via embeddings ONNX.
    """

    def __init__(self, memory_path: Path | str = "memory_store"):
        self.memory_path = Path(memory_path)
        self._notes: dict[str, PassiveNote] = {}   # id → nota (backward compat)
        self._chunks: dict[str, Chunk] = {}         # chunk_id → Chunk

        # ChunkStore: persistencia de chunks en .chunks/
        self._chunk_store = ChunkStore(self.memory_path / ".chunks")

        # Chunker: divide notas en chunks estructurales (sin semantic splitter)
        self._chunker = Chunker(max_chunk_tokens=512)

        self.load_all()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Parsea todas las notas .md del directorio memory_store/ y carga chunks."""
        self._notes.clear()
        self._chunks.clear()

        if not self.memory_path.exists():
            self.memory_path.mkdir(parents=True, exist_ok=True)
            return

        # 1. Cargar notas originales
        for path in self.memory_path.glob("*.md"):
            note = self._parse_note(path)
            if note:
                self._notes[note.id] = note

        # 2. Cargar chunks del ChunkStore (o crear si no existen)
        stored_chunks = self._chunk_store.load_chunks()
        if stored_chunks:
            for chunk in stored_chunks:
                self._chunks[chunk.id] = chunk
        else:
            # Primera ejecución: chunkear todas las notas
            for note in self._notes.values():
                chunks = self._chunker.chunk_note(note)
                for chunk in chunks:
                    self._chunk_store.save_chunk(chunk)
                    self._chunks[chunk.id] = chunk

    def _parse_note(self, path: Path) -> PassiveNote | None:
        try:
            post = frontmatter.load(str(path))
            note_id   = str(post.metadata.get("id", path.stem))
            tags_raw  = post.metadata.get("tags", [])
            tags      = [str(t).lower() for t in tags_raw] if isinstance(tags_raw, list) else []
            priority  = int(post.metadata.get("priority", 5))
            content   = post.content.strip()
            return PassiveNote(
                id=note_id,
                tags=tags,
                priority=priority,
                content=content,
                source_path=path,
            )
        except Exception as e:
            print(f"[PassiveMemory] Error al parsear {path}: {e}")
            return None

    def reload(self) -> None:
        """Recarga las notas desde disco (útil si se crearon notas nuevas en sesión)."""
        self.load_all()

    # ------------------------------------------------------------------
    # Búsqueda (Fase 1: por tags)
    # ------------------------------------------------------------------

    def get_relevant(
        self,
        query: str,
        method: str = "tags",
        max_results: int = 5,
    ) -> list[Chunk]:
        """
        Devuelve chunks relevantes para el query dado.

        Fase 1 — method='tags':
            Extrae palabras del query y las compara contra los tags de cada chunk.
            Los chunks con más coincidencias de tags aparecen primero.
            Empate resuelto por prioridad.

        Fase 2 — method='embeddings':
            Manejado externamente por EmbeddingIndex.search().
            PassiveMemory devuelve [] para que el Harness use el índice semántico.
        """
        if method == "tags":
            return self._search_chunks_by_tags(query, max_results)

        # "embeddings": manejado externamente por EmbeddingIndex.search()
        return []

    def _search_chunks_by_tags(self, query: str, max_results: int) -> list[Chunk]:
        """Búsqueda determinista por tags sobre los chunks."""
        if not self._chunks:
            return []

        # Normalizar: quitar tildes y caracteres especiales para la comparación
        normalized = self._normalize(query)
        # Tokenizar: palabras de ≥2 letras, en minúsculas
        query_words = set(
            w for w in re.findall(r"\b\w{2,}\b", normalized)
        )
        if not query_words:
            # Sin palabras útiles: devolver los chunks de mayor prioridad
            return sorted(self._chunks.values(), key=lambda c: -c.priority)[:max_results]

        scored: list[tuple[int, int, Chunk]] = []
        for chunk in self._chunks.values():
            # También normalizar los tags para comparación justa
            tag_set = set(self._normalize(t) for t in chunk.tags)
            matches = len(query_words & tag_set)
            if matches > 0:
                scored.append((matches, chunk.priority, chunk))

        # Ordenar: más coincidencias primero, luego por prioridad
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [item[2] for item in scored[:max_results]]

    @staticmethod
    def _normalize(text: str) -> str:
        """Quita tildes y normaliza texto para comparación de tags."""
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    # ------------------------------------------------------------------
    # Presupuesto de tokens
    # ------------------------------------------------------------------

    def apply_token_budget(
        self,
        chunks: list[Chunk],
        max_tokens: int,
    ) -> list[Chunk]:
        """
        Selecciona los chunks que caben dentro del presupuesto de tokens.
        Ordena por prioridad (mayor primero) y corta cuando se excede el límite.
        """
        selected: list[Chunk] = []
        used = 0
        for chunk in sorted(chunks, key=lambda c: -c.priority):
            est = chunk.token_estimate()
            if used + est <= max_tokens:
                selected.append(chunk)
                used += est
        return selected

    # ------------------------------------------------------------------
    # Creación de notas (el LLM puede crear notas via Harness)
    # ------------------------------------------------------------------

    def add_note(
        self,
        note_id: str,
        content: str,
        tags: list[str],
        priority: int = 5,
    ) -> PassiveNote:
        """Crea y persiste una nota nueva en memory_store/, luego la chunkea."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        path = self.memory_path / f"{note_id}.md"

        # Serializar con frontmatter
        post = frontmatter.Post(
            content,
            id=note_id,
            tags=tags,
            priority=priority,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        note = PassiveNote(
            id=note_id,
            tags=[t.lower() for t in tags],
            priority=priority,
            content=content,
            source_path=path,
        )
        self._notes[note_id] = note

        # Chunkear la nota y guardar los chunks
        chunks = self._chunker.chunk_note(note)
        for chunk in chunks:
            self._chunk_store.save_chunk(chunk)
            self._chunks[chunk.id] = chunk

        return note

    # ------------------------------------------------------------------
    # Formato para inyectar en el system prompt
    # ------------------------------------------------------------------

    def format_for_prompt(self, chunks: list[Chunk]) -> str:
        """Convierte una lista de chunks a texto para el system prompt."""
        if not chunks:
            return ""
        parts = ["## Contexto relevante de tu memoria\n"]
        for chunk in chunks:
            parts.append(
                f"### {chunk.title} (de {chunk.parent_note_id})\n{chunk.content}\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    @property
    def notes(self) -> dict[str, PassiveNote]:
        return self._notes

    @property
    def chunks(self) -> dict[str, Chunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._notes)

    def __iter__(self) -> Iterator[PassiveNote]:
        return iter(self._notes.values())
