"""
iico_core/db/watcher.py
========================
NoteWatcher: sincroniza una carpeta de notas ``.md`` con :class:`NoteDB`.

Flujo completo de ingestión::

    [carpeta/*.md]
         │
         ▼
    NoteWatcher.sync()
         │
         ├── parse .md frontmatter  (python-frontmatter)
         ├── comparar content_hash  (SHA-256 vs BD)
         │       ├── sin cambios → skip
         │       └── nueva/modificada ─→
         │               ├── NoteDB.upsert_note()
         │               ├── rechunk (delete old + insert new)
         │               └── embed chunks (si EmbeddingIndex disponible)
         └── notas eliminadas → (opcional) delete_note()

El Watcher NO usa inotify/watchdog: escanea la carpeta on-demand.
Esto lo mantiene zero-dependency y funciona en Windows/Linux/macOS.
Para vigilancia continua, llama ``sync()`` desde un loop o scheduler.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter  # python-frontmatter

from .note_db import NoteDB

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SyncReport
# ---------------------------------------------------------------------------

@dataclass
class SyncReport:
    """Resultado de una ejecución de :meth:`NoteWatcher.sync`."""
    new_notes: int = 0
    modified_notes: int = 0
    unchanged_notes: int = 0
    deleted_notes: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def total_processed(self) -> int:
        return self.new_notes + self.modified_notes

    def __str__(self) -> str:
        return (
            f"SyncReport("
            f"new={self.new_notes}, modified={self.modified_notes}, "
            f"unchanged={self.unchanged_notes}, deleted={self.deleted_notes}, "
            f"chunks={self.total_chunks}, embeddings={self.total_embeddings}, "
            f"errors={len(self.errors)}, elapsed={self.elapsed_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# NoteParser (lógica de parseo aislada para testing)
# ---------------------------------------------------------------------------

class NoteParser:
    """Parsea archivos ``.md`` con frontmatter YAML.

    Formato esperado::

        ---
        id: mi-nota
        tags: [tag1, tag2]
        priority: 7
        ---
        # Contenido
        ...

    Todos los campos del frontmatter son opcionales; se usan defaults
    seguros si faltan.
    """

    @staticmethod
    def parse(md_path: Path) -> dict | None:
        """Parsea un ``.md`` y retorna un dict con los campos normalizados.

        Returns:
            Dict con ``id, title, tags, priority, content, source_path``,
            o ``None`` si el archivo no se pudo parsear.
        """
        try:
            post = frontmatter.load(str(md_path))
        except Exception as e:
            logger.warning("[NoteParser] Error leyendo %s: %s", md_path, e)
            return None

        try:
            meta = post.metadata

            # ID: del frontmatter o stem del archivo
            note_id = str(meta.get("id", md_path.stem)).strip()
            if not note_id:
                note_id = md_path.stem

            # Tags: lista de strings en minúsculas
            tags_raw = meta.get("tags", [])
            if isinstance(tags_raw, list):
                tags = [str(t).lower().strip() for t in tags_raw if t]
            elif isinstance(tags_raw, str):
                tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
            else:
                tags = []

            # Priority: 1-10, default 5
            priority = int(meta.get("priority", 5))
            priority = max(1, min(10, priority))

            # Title: primer H1 del cuerpo o el ID
            content = post.content.strip()
            title = NoteParser._extract_title(content) or note_id

            return {
                "id": note_id,
                "title": title,
                "tags": tags,
                "priority": priority,
                "content": content,
                "source_path": md_path,
            }

        except Exception as e:
            logger.warning("[NoteParser] Error parseando %s: %s", md_path, e)
            return None

    @staticmethod
    def _extract_title(content: str) -> str | None:
        """Extrae el primer encabezado H1 (``# Título``) del cuerpo."""
        for line in content.splitlines():
            m = re.match(r"^#\s+(.+)$", line.strip())
            if m:
                return m.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# NoteChunker mínimo (sin dependencia de rag_bench)
# ---------------------------------------------------------------------------

class _SimpleChunker:
    """Chunker mínimo para la Fase 0: 1 chunk por sección H2.

    Cuando los chunkers de rag_bench estén disponibles, el Watcher
    usará el ``ChunkingPipeline`` configurado externamente.
    Se puede reemplazar via ``NoteWatcher.chunker``.
    """

    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def chunk(self, note_id: str, content: str, tags: list[str], priority: int) -> list[dict]:
        """Divide el contenido en chunks por secciones H2.

        Returns:
            Lista de dicts con ``id, title, content, order``.
        """
        sections = self._split_by_h2(content)
        chunks = []
        for order, (title, body) in enumerate(sections):
            if not body.strip():
                continue
            chunk_id = f"{note_id}::{self._slugify(title)}"
            chunks.append({
                "id": chunk_id,
                "title": title,
                "content": body.strip(),
                "order": order,
            })

        # Garantizar al menos 1 chunk (nota sin secciones H2)
        if not chunks:
            chunks.append({
                "id": f"{note_id}::contenido",
                "title": note_id,
                "content": content.strip(),
                "order": 0,
            })
        return chunks

    def _split_by_h2(self, content: str) -> list[tuple[str, str]]:
        """Divide el texto por encabezados ## y retorna (título, cuerpo)."""
        sections: list[tuple[str, str]] = []
        current_title = "introduccion"
        current_lines: list[str] = []

        for line in content.splitlines():
            m = re.match(r"^#{2,3}\s+(.+)$", line)
            if m:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                    current_lines = []
                current_title = m.group(1).strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections if sections else [("contenido", content)]

    @staticmethod
    def _slugify(title: str) -> str:
        slug = title.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "sin-titulo"


# ---------------------------------------------------------------------------
# NoteWatcher
# ---------------------------------------------------------------------------

class NoteWatcher:
    """Sincroniza una carpeta de notas ``.md`` con :class:`NoteDB`.

    Diseñado para ser llamado on-demand (``sync()``), no como daemon.
    Funciona en Windows, Linux y macOS sin dependencias adicionales.

    Args:
        notes_dir: Carpeta con las notas ``.md``.
        db: Instancia de :class:`NoteDB` donde se persistirán las notas.
        chunker: Instancia con método ``chunk(note_id, content, tags, priority) → list[dict]``.
                 Si es ``None``, se usa el :class:`_SimpleChunker` por defecto.
        embedding_index: Instancia de ``EmbeddingIndex`` (opcional).
                         Si se provee, se calculan embeddings para cada chunk.
        delete_removed: Si ``True``, elimina de la BD las notas que ya no
                        existen en la carpeta. Default: ``False`` (conservador).
        chunking_strategy_name: Nombre que se guarda en ``chunks.chunking_strategy``
                                para trazabilidad. Default: ``"structural"``.
    """

    def __init__(
        self,
        notes_dir: Path | str,
        db: NoteDB,
        chunker=None,
        embedding_index=None,
        delete_removed: bool = False,
        chunking_strategy_name: str = "structural",
    ) -> None:
        self.notes_dir = Path(notes_dir)
        self.db = db
        self.chunker = chunker or _SimpleChunker()
        self.embedding_index = embedding_index
        self.delete_removed = delete_removed
        self.chunking_strategy_name = chunking_strategy_name

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def sync(self) -> SyncReport:
        """Escanea la carpeta y sincroniza cambios con la BD.

        Pasos:
        1. Lista todos los ``.md`` en ``notes_dir`` (no recursivo).
        2. Para cada archivo: parsea, compara hash, decide si ingestar.
        3. Opcionalmente elimina notas que ya no existen en disco.

        Returns:
            :class:`SyncReport` con conteos de notas y chunks procesados.
        """
        t0 = time.perf_counter()
        report = SyncReport()

        if not self.notes_dir.exists():
            logger.warning("[NoteWatcher] Carpeta no existe: %s", self.notes_dir)
            report.elapsed_ms = (time.perf_counter() - t0) * 1000
            return report

        # IDs de notas encontradas en disco (para detectar eliminadas)
        disk_ids: set[str] = set()

        for md_path in sorted(self.notes_dir.glob("*.md")):
            parsed = NoteParser.parse(md_path)
            if parsed is None:
                report.errors.append(f"Error parseando: {md_path.name}")
                continue

            note_id = parsed["id"]
            disk_ids.add(note_id)

            # Decidir acción
            content_changed = self.db.note_needs_update(note_id, parsed["content"])
            needs_update = self.db.note_needs_update(note_id, parsed["content"], self.chunking_strategy_name)
            
            if needs_update:
                is_new = self.db.get_note(note_id) is None
                try:
                    chunks_count, emb_count = self._ingest_note(parsed, content_changed)
                    report.total_chunks += chunks_count
                    report.total_embeddings += emb_count
                    if is_new:
                        report.new_notes += 1
                    else:
                        report.modified_notes += 1
                except Exception as e:
                    msg = f"Error ingestando {md_path.name}: {e}"
                    logger.error("[NoteWatcher] %s", msg)
                    report.errors.append(msg)
            else:
                report.unchanged_notes += 1

        # Eliminar notas que ya no están en disco
        if self.delete_removed:
            deleted = self._remove_stale(disk_ids)
            report.deleted_notes = deleted

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("[NoteWatcher] %s", report)
        return report

    def ingest_note(self, md_path: Path | str) -> tuple[int, int]:
        """Ingesta una nota individual, independientemente de su estado.

        Fuerza re-parseo, re-chunking y re-embedding aunque no haya cambios.
        Útil para re-indexar una nota manualmente.

        Args:
            md_path: Ruta al archivo ``.md``.

        Returns:
            Tupla ``(chunks_count, embeddings_count)``.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el archivo no se pudo parsear.
        """
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Nota no encontrada: {md_path}")

        parsed = NoteParser.parse(md_path)
        if parsed is None:
            raise ValueError(f"No se pudo parsear: {md_path}")

        return self._ingest_note(parsed, content_changed=True)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _ingest_note(self, parsed: dict, content_changed: bool) -> tuple[int, int]:
        """Persiste nota, chunks y embeddings en la BD.

        Returns:
            Tupla ``(chunks_count, embeddings_count)``.
        """
        note_id = parsed["id"]

        # 1. Upsert nota
        self.db.upsert_note(
            note_id=note_id,
            title=parsed["title"],
            tags=parsed["tags"],
            priority=parsed["priority"],
            content=parsed["content"],
            source_path=str(parsed["source_path"]),
        )

        # 2. Borrar chunks anteriores
        if content_changed:
            self.db.rechunk_note(note_id)
        else:
            self.db.delete_chunks_for_strategy(note_id, self.chunking_strategy_name)

        # 3. Chunkear con la estrategia configurada
        raw_chunks = self.chunker.chunk(
            note_id=note_id,
            content=parsed["content"],
            tags=parsed["tags"],
            priority=parsed["priority"],
        )

        # 4. Persistir chunks
        for chunk in raw_chunks:
            # Modificamos el ID para evitar colisiones entre estrategias
            new_chunk_id = f"{chunk['id']}::{self.chunking_strategy_name}"
            chunk["id"] = new_chunk_id
            
            self.db.upsert_chunk(
                chunk_id=chunk["id"],
                note_id=note_id,
                title=chunk["title"],
                content=chunk["content"],
                tags=parsed["tags"],
                priority=parsed["priority"],
                order=chunk["order"],
                chunking_strategy=self.chunking_strategy_name,
            )

        chunks_count = len(raw_chunks)
        emb_count = 0

        # 5. Calcular embeddings si hay índice disponible
        if self.embedding_index is not None:
            emb_count = self._embed_chunks(note_id, raw_chunks)

        logger.debug(
            "[NoteWatcher] Ingestada '%s': %d chunks, %d embeddings",
            note_id, chunks_count, emb_count,
        )
        return chunks_count, emb_count

    def _embed_chunks(self, note_id: str, raw_chunks: list[dict]) -> int:
        """Vectoriza chunks y guarda embeddings en la BD.

        Returns:
            Número de embeddings generados.
        """
        count = 0
        for chunk in raw_chunks:
            try:
                vec = self.embedding_index.vectorize(chunk["content"])
                self.db.save_embedding(
                    chunk_id=chunk["id"],
                    vector=vec,
                )
                count += 1
            except Exception as e:
                logger.warning(
                    "[NoteWatcher] Error embediendo chunk '%s': %s",
                    chunk["id"], e,
                )
        return count

    def _remove_stale(self, disk_ids: set[str]) -> int:
        """Elimina de la BD las notas que ya no existen en disco.

        Returns:
            Número de notas eliminadas.
        """
        db_ids = {n["id"] for n in self.db.list_notes()}
        stale = db_ids - disk_ids
        count = 0
        for note_id in stale:
            if self.db.delete_note(note_id):
                logger.info("[NoteWatcher] Eliminada nota stale: %s", note_id)
                count += 1
        return count
