"""
iico_core/db/note_db.py
========================
SQLite ultra-ligera para notas, chunks y embeddings.

Esquema relacional::

    notes ─1:N─→ chunks ─1:1─→ embeddings
    chunks ─N:N─→ chunk_links  (cross-links semánticos)

El archivo ``.db`` vive junto a las notas (ej. ``benchmarks/test_notes/iico.db``)
o en cualquier ruta que el llamador especifique.

Características clave:

* **Zero dependencies**: usa el módulo ``sqlite3`` integrado en Python.
* **WAL mode**: permite lecturas concurrentes (agente + benchmark).
* **CASCADE deletes**: borrar una nota elimina automáticamente sus chunks,
  embeddings y cross-links.
* **Embeddings como BLOB**: 384 floats × 4 bytes = 1.5 KB por chunk,
  todo dentro de la BD.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS notes (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '[]',
    priority     INTEGER NOT NULL DEFAULT 5,
    content      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id                TEXT PRIMARY KEY,
    note_id           TEXT NOT NULL,
    title             TEXT NOT NULL,
    content           TEXT NOT NULL,
    tags              TEXT NOT NULL DEFAULT '[]',
    priority          INTEGER NOT NULL DEFAULT 5,
    "order"           INTEGER NOT NULL DEFAULT 0,
    content_hash      TEXT NOT NULL,
    token_estimate    INTEGER NOT NULL DEFAULT 0,
    chunking_strategy TEXT NOT NULL DEFAULT 'structural',
    FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id   TEXT PRIMARY KEY,
    vector     BLOB NOT NULL,
    model_name TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dim        INTEGER NOT NULL DEFAULT 384,
    created_at TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_links (
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    similarity      REAL NOT NULL,
    link_type       TEXT NOT NULL DEFAULT 'semantic_recurrence',
    detected_by     TEXT NOT NULL DEFAULT 'autocorrelation',
    PRIMARY KEY (source_chunk_id, target_chunk_id),
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_note_id
    ON chunks(note_id);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy
    ON chunks(chunking_strategy);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id
    ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_links_source
    ON chunk_links(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_links_target
    ON chunk_links(target_chunk_id);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Timestamp ISO-8601 en UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    """SHA-256 hex-digest de un string UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convierte un ``sqlite3.Row`` en un dict plano."""
    return dict(row)


def _ensure_numpy() -> None:
    """Lanza ``ImportError`` si numpy no está disponible."""
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy no está instalado. "
            "Instala iico-core[embeddings] para usar embeddings."
        )


# ---------------------------------------------------------------------------
# NoteDB
# ---------------------------------------------------------------------------

class NoteDB:
    """Base de datos SQLite para notas, chunks y embeddings.

    Uso típico::

        db = NoteDB("benchmarks/test_notes/iico.db")
        db.upsert_note("mi_nota", "Mi Nota", ["tag1"], 5, "contenido...", "/ruta/mi_nota.md")
        db.upsert_chunk("mi_nota::intro", "mi_nota", "Intro", "texto...", ["tag1"], 5, 0)
        db.save_embedding("mi_nota::intro", embedding_vector)
        db.close()
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Cierra la conexión a la BD de forma segura."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "NoteDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================================================================
    # NOTAS
    # ==================================================================

    def upsert_note(
        self,
        note_id: str,
        title: str,
        tags: list[str],
        priority: int,
        content: str,
        source_path: str | Path,
    ) -> None:
        """Inserta o actualiza una nota.

        Si la nota ya existe, se actualizan todos los campos y se
        re-calcula el ``content_hash``.  Los chunks asociados **no**
        se borran automáticamente; usar :meth:`rechunk_note` para eso.
        """
        now = _now_iso()
        content_hash = _sha256(content)
        self._conn.execute(
            """
            INSERT INTO notes (id, title, tags, priority, content,
                               source_path, content_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title        = excluded.title,
                tags         = excluded.tags,
                priority     = excluded.priority,
                content      = excluded.content,
                source_path  = excluded.source_path,
                content_hash = excluded.content_hash,
                updated_at   = excluded.updated_at
            """,
            (
                note_id,
                title,
                json.dumps(tags, ensure_ascii=False),
                priority,
                content,
                str(source_path),
                content_hash,
                now,
                now,
            ),
        )
        self._conn.commit()

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        """Devuelve la nota como dict, o ``None`` si no existe."""
        cur = self._conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        d = _row_to_dict(row)
        d["tags"] = json.loads(d["tags"])
        return d

    def list_notes(self) -> list[dict[str, Any]]:
        """Devuelve todas las notas, ordenadas por ``updated_at`` desc."""
        cur = self._conn.execute(
            "SELECT * FROM notes ORDER BY updated_at DESC"
        )
        results = []
        for row in cur.fetchall():
            d = _row_to_dict(row)
            d["tags"] = json.loads(d["tags"])
            results.append(d)
        return results

    def delete_note(self, note_id: str) -> bool:
        """Elimina una nota y todo lo asociado (CASCADE).

        Returns:
            ``True`` si se eliminó algo, ``False`` si la nota no existía.
        """
        cur = self._conn.execute(
            "DELETE FROM notes WHERE id = ?", (note_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def note_needs_update(self, note_id: str, content: str, strategy_name: str | None = None) -> bool:
        """Verifica si el contenido de una nota ha cambiado o si faltan chunks para la estrategia.

        Compara el ``content_hash`` de la nota en disco vs BD.
        Adicionalmente, si se provee ``strategy_name``, verifica si existen chunks
        para esa estrategia específica. Si no hay chunks, se necesita actualización.
        """
        cur = self._conn.execute(
            "SELECT content_hash FROM notes WHERE id = ?", (note_id,)
        )
        row = cur.fetchone()
        if row is None:
            return True
            
        # Si el contenido cambió, necesita update total
        if row["content_hash"] != _sha256(content):
            return True
            
        # Si el contenido es igual, checar si faltan chunks para la estrategia
        if strategy_name is not None:
            cur = self._conn.execute(
                "SELECT 1 FROM chunks WHERE note_id = ? AND chunking_strategy = ? LIMIT 1",
                (note_id, strategy_name)
            )
            if cur.fetchone() is None:
                return True
                
        return False

    def get_content_hash(self, note_id: str) -> str | None:
        """Devuelve el ``content_hash`` almacenado, o ``None``."""
        cur = self._conn.execute(
            "SELECT content_hash FROM notes WHERE id = ?", (note_id,)
        )
        row = cur.fetchone()
        return row["content_hash"] if row else None

    # ==================================================================
    # CHUNKS
    # ==================================================================

    def upsert_chunk(
        self,
        chunk_id: str,
        note_id: str,
        title: str,
        content: str,
        tags: list[str],
        priority: int,
        order: int,
        chunking_strategy: str = "structural",
    ) -> None:
        """Inserta o actualiza un chunk."""
        content_hash = _sha256(content)
        token_estimate = len(content) // 4
        self._conn.execute(
            """
            INSERT INTO chunks (id, note_id, title, content, tags, priority,
                                "order", content_hash, token_estimate,
                                chunking_strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                note_id           = excluded.note_id,
                title             = excluded.title,
                content           = excluded.content,
                tags              = excluded.tags,
                priority          = excluded.priority,
                "order"           = excluded."order",
                content_hash      = excluded.content_hash,
                token_estimate    = excluded.token_estimate,
                chunking_strategy = excluded.chunking_strategy
            """,
            (
                chunk_id,
                note_id,
                title,
                content,
                json.dumps(tags, ensure_ascii=False),
                priority,
                order,
                content_hash,
                token_estimate,
                chunking_strategy,
            ),
        )
        self._conn.commit()

    def get_chunks_for_note(self, note_id: str) -> list[dict[str, Any]]:
        """Devuelve chunks de una nota, ordenados por ``order``."""
        cur = self._conn.execute(
            'SELECT * FROM chunks WHERE note_id = ? ORDER BY "order"',
            (note_id,),
        )
        return [self._parse_chunk_row(row) for row in cur.fetchall()]

    def get_all_chunks(self, strategy_name: str | None = None) -> list[dict[str, Any]]:
        """Devuelve todos los chunks de todas las notas.
        
        Si ``strategy_name`` es provisto, filtra los chunks por esa estrategia.
        """
        if strategy_name is not None:
            cur = self._conn.execute(
                "SELECT * FROM chunks WHERE chunking_strategy = ? ORDER BY note_id, \"order\"",
                (strategy_name,)
            )
        else:
            cur = self._conn.execute("SELECT * FROM chunks ORDER BY note_id, \"order\"")
        results = []
        for row in cur.fetchall():
            results.append(self._parse_chunk_row(row))
        return results

    def delete_chunks_for_note(self, note_id: str) -> int:
        """Elimina todos los chunks (todas las estrategias) de una nota."""
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE note_id = ?", (note_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def delete_chunks_for_strategy(self, note_id: str, strategy_name: str) -> int:
        """Elimina los chunks de una nota específicos a una estrategia."""
        cur = self._conn.execute(
            "DELETE FROM chunks WHERE note_id = ? AND chunking_strategy = ?",
            (note_id, strategy_name)
        )
        self._conn.commit()
        return cur.rowcount

    def rechunk_note(self, note_id: str) -> int:
        """Borra los chunks existentes de una nota para re-chunkear.

        Alias semántico de :meth:`delete_chunks_for_note` — señaliza
        la intención de que se va a re-chunkear la nota inmediatamente.

        Returns:
            Número de chunks eliminados.
        """
        return self.delete_chunks_for_note(note_id)

    @staticmethod
    def _parse_chunk_row(row: sqlite3.Row) -> dict[str, Any]:
        d = _row_to_dict(row)
        d["tags"] = json.loads(d["tags"])
        return d

    # ==================================================================
    # EMBEDDINGS
    # ==================================================================

    def save_embedding(
        self,
        chunk_id: str,
        vector: Any,  # np.ndarray
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Guarda un embedding como BLOB.

        El vector se serializa con ``ndarray.tobytes()`` y se almacena
        directamente en la tabla ``embeddings``.  Al cargarlo, se
        reconstruye con ``np.frombuffer(blob, dtype=np.float32)``.

        Args:
            chunk_id: ID del chunk al que pertenece.
            vector: numpy array de shape ``(dim,)``, dtype ``float32``.
            model_name: nombre del modelo que generó el embedding.
        """
        _ensure_numpy()
        vec = np.asarray(vector, dtype=np.float32)
        dim = vec.shape[0]
        blob = vec.tobytes()
        now = _now_iso()
        self._conn.execute(
            """
            INSERT INTO embeddings (chunk_id, vector, model_name, dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                vector     = excluded.vector,
                model_name = excluded.model_name,
                dim        = excluded.dim,
                created_at = excluded.created_at
            """,
            (chunk_id, blob, model_name, dim, now),
        )
        self._conn.commit()

    def load_embedding(self, chunk_id: str) -> Any | None:
        """Carga el embedding de un chunk como ``np.ndarray``.

        Returns:
            ``np.ndarray`` de shape ``(dim,)`` y dtype ``float32``,
            o ``None`` si no existe.
        """
        _ensure_numpy()
        cur = self._conn.execute(
            "SELECT vector, dim FROM embeddings WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32).copy()

    def load_all_embeddings(self, strategy_name: str | None = None) -> dict[str, Any]:
        """Carga todos los embeddings como ``{chunk_id: np.ndarray}``.

        Esto permite construir la matriz de embeddings en RAM para
        cosine similarity brute-force.
        Si se provee ``strategy_name``, sólo carga los embeddings de los
        chunks generados por esa estrategia.
        """
        _ensure_numpy()
        if strategy_name is not None:
            cur = self._conn.execute(
                """
                SELECT e.chunk_id, e.vector 
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                WHERE c.chunking_strategy = ?
                """,
                (strategy_name,)
            )
        else:
            cur = self._conn.execute("SELECT chunk_id, vector FROM embeddings")
            
        result: dict[str, Any] = {}
        for row in cur.fetchall():
            vec = np.frombuffer(row["vector"], dtype=np.float32).copy()
            result[row["chunk_id"]] = vec
        return result

    def has_embedding(self, chunk_id: str) -> bool:
        """Verifica si un chunk tiene embedding."""
        cur = self._conn.execute(
            "SELECT 1 FROM embeddings WHERE chunk_id = ?", (chunk_id,)
        )
        return cur.fetchone() is not None

    def chunks_without_embeddings(self) -> list[str]:
        """Devuelve IDs de chunks que no tienen embedding calculado."""
        cur = self._conn.execute(
            """
            SELECT c.id FROM chunks c
            LEFT JOIN embeddings e ON c.id = e.chunk_id
            WHERE e.chunk_id IS NULL
            """
        )
        return [row["id"] for row in cur.fetchall()]

    def delete_embedding(self, chunk_id: str) -> bool:
        """Elimina el embedding de un chunk.

        Returns:
            ``True`` si se eliminó, ``False`` si no existía.
        """
        cur = self._conn.execute(
            "DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ==================================================================
    # CROSS-LINKS
    # ==================================================================

    def save_link(
        self,
        source_chunk_id: str,
        target_chunk_id: str,
        similarity: float,
        link_type: str = "semantic_recurrence",
        detected_by: str = "autocorrelation",
    ) -> None:
        """Guarda un cross-link semántico entre dos chunks.

        El link es direccional: ``source → target``.  Si ya existe,
        se actualiza la similaridad.
        """
        self._conn.execute(
            """
            INSERT INTO chunk_links (source_chunk_id, target_chunk_id,
                                     similarity, link_type, detected_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_chunk_id, target_chunk_id) DO UPDATE SET
                similarity  = excluded.similarity,
                link_type   = excluded.link_type,
                detected_by = excluded.detected_by
            """,
            (source_chunk_id, target_chunk_id, similarity, link_type, detected_by),
        )
        self._conn.commit()

    def save_links_batch(
        self,
        links: list[tuple[str, str, float, str, str]],
    ) -> int:
        """Guarda múltiples cross-links en una sola transacción.

        Cada tupla: ``(source_id, target_id, similarity, link_type, detected_by)``.

        Returns:
            Número de links insertados/actualizados.
        """
        self._conn.executemany(
            """
            INSERT INTO chunk_links (source_chunk_id, target_chunk_id,
                                     similarity, link_type, detected_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_chunk_id, target_chunk_id) DO UPDATE SET
                similarity  = excluded.similarity,
                link_type   = excluded.link_type,
                detected_by = excluded.detected_by
            """,
            links,
        )
        self._conn.commit()
        return len(links)

    def get_links_for_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        """Devuelve todos los links donde ``chunk_id`` es source o target."""
        cur = self._conn.execute(
            """
            SELECT * FROM chunk_links
            WHERE source_chunk_id = ? OR target_chunk_id = ?
            ORDER BY similarity DESC
            """,
            (chunk_id, chunk_id),
        )
        return [_row_to_dict(row) for row in cur.fetchall()]

    def get_outgoing_links(self, chunk_id: str) -> list[dict[str, Any]]:
        """Devuelve links salientes (source = chunk_id)."""
        cur = self._conn.execute(
            """
            SELECT * FROM chunk_links
            WHERE source_chunk_id = ?
            ORDER BY similarity DESC
            """,
            (chunk_id,),
        )
        return [_row_to_dict(row) for row in cur.fetchall()]

    def get_linked_chunks(self, chunk_id: str) -> list[dict[str, Any]]:
        """Devuelve chunks linkeados semánticamente a ``chunk_id``.

        Busca en ambas direcciones (source → target y target → source)
        y devuelve los chunks conectados con su similaridad.
        """
        cur = self._conn.execute(
            """
            SELECT c.*, cl.similarity, cl.link_type
            FROM chunk_links cl
            JOIN chunks c ON (
                (cl.target_chunk_id = c.id AND cl.source_chunk_id = ?)
                OR
                (cl.source_chunk_id = c.id AND cl.target_chunk_id = ?)
            )
            ORDER BY cl.similarity DESC
            """,
            (chunk_id, chunk_id),
        )
        results = []
        for row in cur.fetchall():
            d = _row_to_dict(row)
            d["tags"] = json.loads(d["tags"])
            results.append(d)
        return results

    def delete_links_for_note(self, note_id: str) -> int:
        """Elimina todos los cross-links cuyos chunks pertenecen a ``note_id``.

        Returns:
            Número de links eliminados.
        """
        cur = self._conn.execute(
            """
            DELETE FROM chunk_links
            WHERE source_chunk_id IN (SELECT id FROM chunks WHERE note_id = ?)
               OR target_chunk_id IN (SELECT id FROM chunks WHERE note_id = ?)
            """,
            (note_id, note_id),
        )
        self._conn.commit()
        return cur.rowcount

    # ==================================================================
    # BÚSQUEDA DIRECTA
    # ==================================================================

    def search_by_tags(
        self,
        query_tags: set[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Busca chunks cuyo JSON ``tags`` contenga alguno de ``query_tags``.

        Ordena por número de coincidencias (desc) y luego prioridad (desc).

        Nota: Usa ``LIKE`` por simplicidad (suficiente para ~1000 chunks).
        Para datasets grandes, considerar FTS5.
        """
        if not query_tags:
            return []

        # Construir condición OR con LIKE para cada tag
        conditions = []
        params: list[str] = []
        for tag in query_tags:
            conditions.append('tags LIKE ?')
            params.append(f'%"{tag}"%')

        where_clause = " OR ".join(conditions)
        query = f"""
            SELECT *, ({" + ".join(f"(tags LIKE ?)" for _ in query_tags)}) as match_count
            FROM chunks
            WHERE {where_clause}
            ORDER BY match_count DESC, priority DESC
            LIMIT ?
        """
        # Parámetros: primero los del SELECT count, luego los del WHERE
        all_params = list(params) + list(params) + [str(top_k)]
        cur = self._conn.execute(query, all_params)
        return [self._parse_chunk_row(row) for row in cur.fetchall()]

    # ==================================================================
    # UTILIDADES
    # ==================================================================

    def stats(self) -> dict[str, int]:
        """Estadísticas de la BD: conteo de notas, chunks, embeddings, links."""
        result = {}
        for table in ("notes", "chunks", "embeddings", "chunk_links"):
            cur = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {table}")  # noqa: S608
            result[table] = cur.fetchone()["cnt"]
        return result

    def clear(self) -> None:
        """Elimina **todos** los datos de la BD (útil para re-benchmarks)."""
        self._conn.executescript("""
            DELETE FROM chunk_links;
            DELETE FROM embeddings;
            DELETE FROM chunks;
            DELETE FROM notes;
        """)
        self._conn.commit()

    def vacuum(self) -> None:
        """Compacta el archivo ``.db`` después de borrados masivos."""
        self._conn.execute("VACUUM")

    @property
    def connection(self) -> sqlite3.Connection:
        """Acceso directo a la conexión SQLite (para queries custom)."""
        return self._conn
