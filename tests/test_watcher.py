"""
tests/test_watcher.py
======================
Tests para NoteWatcher y NoteParser.

Ejecutar::

    pytest tests/test_watcher.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from iico_core.db.note_db import NoteDB
from iico_core.db.watcher import NoteParser, NoteWatcher, SyncReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> NoteDB:
    db_path = tmp_path / "test.db"
    _db = NoteDB(db_path)
    yield _db
    _db.close()


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    """Carpeta vacía de notas."""
    d = tmp_path / "notes"
    d.mkdir()
    return d


def write_note(path: Path, content: str) -> None:
    """Helper: escribe una nota .md en el directorio."""
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ===========================================================================
# Tests de NoteParser
# ===========================================================================

class TestNoteParser:
    def test_full_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "nota_completa.md"
        write_note(f, """\
            ---
            id: mi-nota
            tags: [splay, cache, arquitectura]
            priority: 8
            ---
            # Mi Nota
            Contenido de la nota con múltiples párrafos.
        """)
        result = NoteParser.parse(f)
        assert result is not None
        assert result["id"] == "mi-nota"
        assert result["tags"] == ["splay", "cache", "arquitectura"]
        assert result["priority"] == 8
        assert result["title"] == "Mi Nota"
        assert "Contenido" in result["content"]

    def test_id_fallback_to_stem(self, tmp_path: Path) -> None:
        f = tmp_path / "sin_id.md"
        write_note(f, """\
            ---
            tags: [test]
            ---
            Contenido sin ID en frontmatter.
        """)
        result = NoteParser.parse(f)
        assert result is not None
        assert result["id"] == "sin_id"

    def test_empty_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "minima.md"
        write_note(f, """\
            ---
            ---
            # Solo contenido
            Sin tags ni prioridad.
        """)
        result = NoteParser.parse(f)
        assert result is not None
        assert result["tags"] == []
        assert result["priority"] == 5  # default

    def test_priority_clamped(self, tmp_path: Path) -> None:
        f = tmp_path / "extrema.md"
        write_note(f, """\
            ---
            priority: 99
            ---
            Contenido.
        """)
        result = NoteParser.parse(f)
        assert result["priority"] == 10  # clamped a max

    def test_tags_normalized_lowercase(self, tmp_path: Path) -> None:
        f = tmp_path / "tags.md"
        write_note(f, """\
            ---
            tags: [SPlay, CACHE, Arquitectura]
            ---
            Contenido.
        """)
        result = NoteParser.parse(f)
        assert result["tags"] == ["splay", "cache", "arquitectura"]

    def test_extract_title_h1(self, tmp_path: Path) -> None:
        f = tmp_path / "titulo.md"
        write_note(f, """\
            ---
            id: t1
            ---
            # Este es el título

            Contenido bajo el título.
        """)
        result = NoteParser.parse(f)
        assert result["title"] == "Este es el título"

    def test_title_fallback_to_id(self, tmp_path: Path) -> None:
        f = tmp_path / "sin_h1.md"
        write_note(f, """\
            ---
            id: sin-h1
            ---
            Solo párrafos sin H1.
        """)
        result = NoteParser.parse(f)
        assert result["title"] == "sin-h1"

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        f = tmp_path / "no_existe.md"
        result = NoteParser.parse(f)
        assert result is None

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad_yaml.md"
        # python-frontmatter es bastante tolerante, pero testeamos que no crashea
        f.write_text("---\nbad: [yaml: here\n---\nContenido.", encoding="utf-8")
        # Puede retornar None o un dict degradado — no debe lanzar excepción
        result = NoteParser.parse(f)
        # No importa el resultado, solo que no lanzó excepción


# ===========================================================================
# Tests de NoteWatcher — sync básico
# ===========================================================================

class TestNoteWatcherSync:
    def test_sync_empty_dir(self, db: NoteDB, notes_dir: Path) -> None:
        watcher = NoteWatcher(notes_dir, db)
        report = watcher.sync()
        assert report.new_notes == 0
        assert report.unchanged_notes == 0
        assert report.errors == []

    def test_sync_new_note(self, db: NoteDB, notes_dir: Path) -> None:
        write_note(notes_dir / "nota1.md", """\
            ---
            id: nota1
            tags: [test]
            priority: 7
            ---
            # Nota 1

            ## Sección A
            Contenido de la sección A.

            ## Sección B
            Contenido de la sección B.
        """)
        watcher = NoteWatcher(notes_dir, db)
        report = watcher.sync()

        assert report.new_notes == 1
        assert report.modified_notes == 0
        assert report.unchanged_notes == 0
        assert report.errors == []
        assert report.total_chunks >= 1

        # Verificar BD
        note = db.get_note("nota1")
        assert note is not None
        assert note["title"] == "Nota 1"
        assert note["tags"] == ["test"]
        chunks = db.get_chunks_for_note("nota1")
        assert len(chunks) >= 1

    def test_sync_two_notes(self, db: NoteDB, notes_dir: Path) -> None:
        for i in range(1, 3):
            write_note(notes_dir / f"nota{i}.md", f"""\
                ---
                id: nota{i}
                tags: [tag{i}]
                ---
                # Nota {i}
                Contenido {i}.
            """)
        watcher = NoteWatcher(notes_dir, db)
        report = watcher.sync()
        assert report.new_notes == 2
        assert db.stats()["notes"] == 2

    def test_sync_unchanged_note_skipped(self, db: NoteDB, notes_dir: Path) -> None:
        write_note(notes_dir / "nota1.md", """\
            ---
            id: nota1
            ---
            # Nota
            Contenido fijo.
        """)
        watcher = NoteWatcher(notes_dir, db)
        watcher.sync()  # Primera pasada
        report = watcher.sync()  # Segunda pasada sin cambios

        assert report.new_notes == 0
        assert report.modified_notes == 0
        assert report.unchanged_notes == 1

    def test_sync_modified_note(self, db: NoteDB, notes_dir: Path) -> None:
        f = notes_dir / "nota1.md"
        write_note(f, """\
            ---
            id: nota1
            ---
            Contenido v1.
        """)
        watcher = NoteWatcher(notes_dir, db)
        watcher.sync()

        # Modificar la nota
        write_note(f, """\
            ---
            id: nota1
            ---
            Contenido v2, diferente al anterior.
        """)
        report = watcher.sync()
        assert report.modified_notes == 1

        # Verificar que el contenido se actualizó
        note = db.get_note("nota1")
        assert "v2" in note["content"]

    def test_sync_elapsed_time_positive(self, db: NoteDB, notes_dir: Path) -> None:
        watcher = NoteWatcher(notes_dir, db)
        report = watcher.sync()
        assert report.elapsed_ms >= 0.0


# ===========================================================================
# Tests de NoteWatcher — delete_removed
# ===========================================================================

class TestNoteWatcherDeleteRemoved:
    def test_stale_notes_not_deleted_by_default(
        self, db: NoteDB, notes_dir: Path
    ) -> None:
        write_note(notes_dir / "nota1.md", """\
            ---
            id: nota1
            ---
            Contenido.
        """)
        watcher = NoteWatcher(notes_dir, db)
        watcher.sync()

        # Eliminar el archivo
        (notes_dir / "nota1.md").unlink()
        report = watcher.sync()

        # Por default, no se elimina de la BD
        assert report.deleted_notes == 0
        assert db.get_note("nota1") is not None

    def test_stale_notes_deleted_when_enabled(
        self, db: NoteDB, notes_dir: Path
    ) -> None:
        write_note(notes_dir / "nota1.md", """\
            ---
            id: nota1
            ---
            Contenido.
        """)
        watcher = NoteWatcher(notes_dir, db, delete_removed=True)
        watcher.sync()

        # Eliminar el archivo
        (notes_dir / "nota1.md").unlink()
        report = watcher.sync()

        assert report.deleted_notes == 1
        assert db.get_note("nota1") is None


# ===========================================================================
# Tests de NoteWatcher — ingest_note individual
# ===========================================================================

class TestNoteWatcherIngestNote:
    def test_ingest_single_note(self, db: NoteDB, notes_dir: Path) -> None:
        f = notes_dir / "individual.md"
        write_note(f, """\
            ---
            id: individual
            tags: [solo]
            priority: 9
            ---
            # Individual
            Nota ingestada individualmente.
        """)
        watcher = NoteWatcher(notes_dir, db)
        chunks, embs = watcher.ingest_note(f)
        assert chunks >= 1
        assert embs == 0  # sin embedding_index

        note = db.get_note("individual")
        assert note is not None
        assert note["priority"] == 9

    def test_ingest_nonexistent_raises(
        self, db: NoteDB, notes_dir: Path
    ) -> None:
        watcher = NoteWatcher(notes_dir, db)
        with pytest.raises(FileNotFoundError):
            watcher.ingest_note(notes_dir / "no_existe.md")

    def test_ingest_force_reindex(self, db: NoteDB, notes_dir: Path) -> None:
        """Ingest forzado re-chunkea aunque no haya cambios."""
        f = notes_dir / "nota1.md"
        write_note(f, """\
            ---
            id: nota1
            ---
            # Nota

            ## Sec A
            Contenido A.
        """)
        watcher = NoteWatcher(notes_dir, db)
        watcher.sync()
        chunks_before = len(db.get_chunks_for_note("nota1"))

        # Ingestar de nuevo (forzado)
        watcher.ingest_note(f)
        chunks_after = len(db.get_chunks_for_note("nota1"))

        # El número de chunks debe ser el mismo (misma nota)
        assert chunks_before == chunks_after


# ===========================================================================
# Tests de rechunking limpio
# ===========================================================================

class TestRechunking:
    def test_rechunk_replaces_old_chunks(
        self, db: NoteDB, notes_dir: Path
    ) -> None:
        f = notes_dir / "nota1.md"
        write_note(f, """\
            ---
            id: nota1
            ---
            # Nota

            ## Sección A
            Primera versión con dos secciones.

            ## Sección B
            Segunda sección.
        """)
        watcher = NoteWatcher(notes_dir, db)
        watcher.sync()
        chunks_v1 = len(db.get_chunks_for_note("nota1"))
        assert chunks_v1 >= 1

        # Modificar a nota con menos secciones
        write_note(f, """\
            ---
            id: nota1
            ---
            # Nota simplificada
            Solo un párrafo sin secciones H2.
        """)
        watcher.sync()
        chunks_v2 = len(db.get_chunks_for_note("nota1"))

        # Debe haber exactamente los chunks de la versión nueva (no acumulados)
        assert chunks_v2 >= 1
        # No quedan chunks "fantasma" de la versión anterior
        all_chunks = db.get_chunks_for_note("nota1")
        for chunk in all_chunks:
            assert "primera versión" not in chunk["content"].lower()


# ===========================================================================
# Tests del chunking_strategy_name
# ===========================================================================

class TestChunkingStrategyName:
    def test_strategy_name_stored(self, db: NoteDB, notes_dir: Path) -> None:
        write_note(notes_dir / "nota1.md", """\
            ---
            id: nota1
            ---
            Contenido de prueba.
        """)
        watcher = NoteWatcher(
            notes_dir, db, chunking_strategy_name="convolution"
        )
        watcher.sync()
        chunks = db.get_chunks_for_note("nota1")
        assert all(c["chunking_strategy"] == "convolution" for c in chunks)


# ===========================================================================
# Test de SyncReport __str__
# ===========================================================================

class TestSyncReport:
    def test_str_representation(self) -> None:
        r = SyncReport(new_notes=2, total_chunks=5, elapsed_ms=12.3)
        s = str(r)
        assert "new=2" in s
        assert "chunks=5" in s

    def test_total_processed(self) -> None:
        r = SyncReport(new_notes=3, modified_notes=2)
        assert r.total_processed == 5
