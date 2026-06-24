"""
iico_core/memory/chunk_store.py
===============================
Almacenamiento de chunks de memoria particionada.

Cada chunk es un fragmento de una nota original, guardado como .md con frontmatter
YAML más un embedding opcional en formato .npy. La estructura en disco es:

    {chunks_root}/
        {parent_note_id}/
            {chunk_id}.md
            {chunk_id}.npy       (opcional, embedding vectorial)

El ChunkStore gestiona lectura/escritura de chunks y detección de cambios
en las notas fuente para re-chunking selectivo.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from ..types import Chunk

if TYPE_CHECKING:
    from .passive import PassiveNote

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------

class ChunkStore:
    """Gestor de persistencia para chunks de memoria particionada.

    Los chunks se almacenan en {chunks_root}/{parent_note_id}/{chunk_id}.md
    con embedding opcional en {chunk_id}.npy.
    """

    def __init__(self, chunks_root: Path | str = "memory_store/.chunks"):
        self.chunks_root = Path(chunks_root)
        self.chunks_root.mkdir(parents=True, exist_ok=True)
        self._embedding_cache: dict[str, "np.ndarray"] = {}

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_filename(chunk_id: str) -> str:
        """Convierte un chunk.id en un nombre de archivo seguro (Windows/Linux).

        Reemplaza ``::`` por ``--`` para evitar caracteres inválidos en Windows.
        El id original se preserva en el frontmatter YAML.
        """
        return chunk_id.replace("::", "--")

    def save_chunk(self, chunk: Chunk) -> None:
        """Persiste el chunk como .md con frontmatter YAML.

        El embedding .npy debe guardarse aparte (ej. via np.save) cuando
        el EmbeddingIndex compute el vector. chunk.embedding_path indica
        dónde se espera encontrar el .npy.
        """
        note_dir = self.chunks_root / chunk.parent_note_id
        note_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._safe_filename(chunk.id)
        md_path = note_dir / f"{safe_name}.md"
        post = frontmatter.Post(
            chunk.content,
            id=chunk.id,
            parent_note_id=chunk.parent_note_id,
            title=chunk.title,
            tags=chunk.tags,
            priority=chunk.priority,
            order=chunk.order,
            content_hash=chunk.content_hash,
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        # Actualizar source_path por si se creó desde cero
        chunk.source_path = md_path

    def load_chunks(self) -> list[Chunk]:
        """Escanea .chunks/ y carga metadata de todos los .md (sin numpy)."""
        chunks: list[Chunk] = []
        if not self.chunks_root.exists():
            return chunks

        for note_dir in sorted(self.chunks_root.iterdir()):
            if not note_dir.is_dir():
                continue
            for md_path in sorted(note_dir.glob("*.md")):
                chunk = self._parse_chunk_md(md_path)
                if chunk is not None:
                    chunks.append(chunk)
        return chunks

    def _parse_chunk_md(self, md_path: Path) -> Chunk | None:
        """Parsea un .md de chunk y devuelve la dataclass Chunk."""
        try:
            post = frontmatter.load(str(md_path))
            meta = post.metadata

            npy_path = md_path.with_suffix(".npy")

            return Chunk(
                id=str(meta.get("id", md_path.stem)),
                parent_note_id=str(meta.get("parent_note_id", "")),
                title=str(meta.get("title", "")),
                content=post.content.strip(),
                tags=list(meta.get("tags", [])),
                priority=int(meta.get("priority", 5)),
                order=int(meta.get("order", 0)),
                source_path=md_path,
                embedding_path=npy_path if npy_path.exists() else None,
                content_hash=str(meta.get("content_hash", "")),
            )
        except Exception as e:
            print(f"[ChunkStore] Error al parsear {md_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_embedding(self, chunk: Chunk) -> "np.ndarray":
        """Carga el embedding .npy del chunk, con caché en RAM.

        Raises:
            ImportError: si numpy no está instalado.
            FileNotFoundError: si no existe el .npy para este chunk.
        """
        if np is None:
            raise ImportError(
                "numpy no está instalado. Instala iico-core[embeddings] "
                "para usar búsqueda semántica."
            )
        if chunk.id in self._embedding_cache:
            return self._embedding_cache[chunk.id]

        if chunk.embedding_path is None or not chunk.embedding_path.exists():
            raise FileNotFoundError(
                f"No se encontró embedding .npy para el chunk {chunk.id}"
            )

        arr = np.load(str(chunk.embedding_path))
        self._embedding_cache[chunk.id] = arr
        return arr

    # ------------------------------------------------------------------
    # Eliminación
    # ------------------------------------------------------------------

    def delete_chunk(self, chunk: Chunk) -> None:
        """Elimina el .md y el .npy del chunk del disco y la caché."""
        if chunk.source_path is not None and chunk.source_path.exists():
            chunk.source_path.unlink()

        if chunk.embedding_path is not None and chunk.embedding_path.exists():
            chunk.embedding_path.unlink()

        self._embedding_cache.pop(chunk.id, None)

    # ------------------------------------------------------------------
    # Detección de cambios
    # ------------------------------------------------------------------

    def detect_changes(self, note_path: Path) -> str:
        """Compara el hash del contenido de la nota fuente contra los chunks.

        Carga el contenido de la nota (cuerpo Markdown, sin frontmatter) y
        compara su SHA-256 contra el content_hash de los chunks derivados.

        Args:
            note_path: Ruta al .md de la nota original en memory_store/.

        Returns:
            "new":       no hay chunks para esta nota → se necesita chunking.
            "modified":  el contenido de la nota difiere de los chunks.
            "unchanged": los chunks están al día.

        Raises:
            FileNotFoundError: si note_path no existe.
        """
        if not note_path.exists():
            raise FileNotFoundError(f"Nota no encontrada: {note_path}")

        # Hash del contenido del .md (cuerpo, sin frontmatter)
        post = frontmatter.load(str(note_path))
        source_hash = hashlib.sha256(
            post.content.strip().encode("utf-8")
        ).hexdigest()

        note_id = note_path.stem
        chunks = self._chunks_for_note(note_id)

        if not chunks:
            return "new"

        # Todos los chunks de una misma nota comparten content_hash
        stored_hash = chunks[0].content_hash
        if source_hash != stored_hash:
            return "modified"
        return "unchanged"

    def _chunks_for_note(self, note_id: str) -> list[Chunk]:
        """Devuelve los chunks cuyo parent_note_id coincide con note_id."""
        note_dir = self.chunks_root / note_id
        if not note_dir.is_dir():
            return []
        chunks: list[Chunk] = []
        for md_path in sorted(note_dir.glob("*.md")):
            chunk = self._parse_chunk_md(md_path)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # Reconstrucción desde notas (placeholder Fase 4 → Chunker)
    # ------------------------------------------------------------------

    def rebuild_from_notes(self, notes: "list[PassiveNote]") -> list[Chunk]:
        """Crea 1 chunk por nota con el contenido completo (placeholder).

        En la Fase 4 real, el Chunker dividirá cada nota en múltiples chunks
        semánticos (uno por sección o párrafo). Por ahora, cada nota se
        convierte en un único chunk con el contenido íntegro.
        """
        chunks: list[Chunk] = []
        for note in notes:
            content = note.content
            content_hash_val = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()

            chunk = Chunk(
                id=note.id,
                parent_note_id=note.id,
                title=note.id,
                content=content,
                tags=list(note.tags),
                priority=note.priority,
                order=0,
                source_path=None,
                embedding_path=None,
                content_hash=content_hash_val,
            )
            chunks.append(chunk)
        return chunks
