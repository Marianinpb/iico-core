"""
tests/test_note_db.py
======================
Tests para NoteDB: CRUD de notas, chunks, embeddings, cross-links y cascades.

Ejecutar::

    pytest tests/test_note_db.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from iico_core.db.note_db import NoteDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

skip_no_numpy = pytest.mark.skipif(
    not _HAS_NUMPY, reason="numpy not installed"
)


@pytest.fixture
def db(tmp_path: Path) -> NoteDB:
    """Crea una NoteDB temporal para cada test."""
    db_path = tmp_path / "test.db"
    _db = NoteDB(db_path)
    yield _db
    _db.close()


@pytest.fixture
def populated_db(db: NoteDB) -> NoteDB:
    """NoteDB con datos de ejemplo precargados."""
    # 2 notas
    db.upsert_note(
        "nota_a", "Nota A", ["splay", "cache"], 8,
        "Contenido de la nota A sobre splay trees y cache.",
        "/fake/nota_a.md",
    )
    db.upsert_note(
        "nota_b", "Nota B", ["embedding", "onnx"], 5,
        "Contenido de la nota B sobre embeddings ONNX.",
        "/fake/nota_b.md",
    )

    # 3 chunks para nota_a
    db.upsert_chunk(
        "nota_a::intro", "nota_a", "Introducción", "Intro al splay tree.",
        ["splay"], 8, 0,
    )
    db.upsert_chunk(
        "nota_a::rotaciones", "nota_a", "Rotaciones", "Zig, zig-zag, zig-zig.",
        ["splay", "cache"], 8, 1,
    )
    db.upsert_chunk(
        "nota_a::metricas", "nota_a", "Métricas", "Convergencia y hit rate.",
        ["cache", "metricas"], 8, 2,
    )

    # 2 chunks para nota_b
    db.upsert_chunk(
        "nota_b::modelo", "nota_b", "Modelo", "all-MiniLM-L6-v2 ONNX.",
        ["embedding", "onnx"], 5, 0,
    )
    db.upsert_chunk(
        "nota_b::cosine", "nota_b", "Cosine Sim", "Similaridad coseno normalizada.",
        ["embedding"], 5, 1,
    )

    return db


# ===========================================================================
# Tests de Notas
# ===========================================================================

class TestNotes:
    def test_upsert_and_get(self, db: NoteDB) -> None:
        db.upsert_note(
            "test", "Test Note", ["tag1", "tag2"], 7,
            "Contenido de prueba.", "/fake/test.md",
        )
        note = db.get_note("test")
        assert note is not None
        assert note["id"] == "test"
        assert note["title"] == "Test Note"
        assert note["tags"] == ["tag1", "tag2"]
        assert note["priority"] == 7
        assert note["content"] == "Contenido de prueba."

    def test_get_nonexistent(self, db: NoteDB) -> None:
        assert db.get_note("no_existe") is None

    def test_upsert_updates_existing(self, db: NoteDB) -> None:
        db.upsert_note("n1", "V1", ["a"], 5, "original", "/n1.md")
        db.upsert_note("n1", "V2", ["b"], 9, "actualizado", "/n1.md")
        note = db.get_note("n1")
        assert note["title"] == "V2"
        assert note["tags"] == ["b"]
        assert note["priority"] == 9
        assert note["content"] == "actualizado"

    def test_list_notes(self, populated_db: NoteDB) -> None:
        notes = populated_db.list_notes()
        assert len(notes) == 2
        ids = {n["id"] for n in notes}
        assert ids == {"nota_a", "nota_b"}

    def test_delete_note(self, populated_db: NoteDB) -> None:
        assert populated_db.delete_note("nota_a") is True
        assert populated_db.get_note("nota_a") is None
        # Verificar que la otra nota sigue
        assert populated_db.get_note("nota_b") is not None

    def test_delete_nonexistent(self, db: NoteDB) -> None:
        assert db.delete_note("fantasma") is False

    def test_note_needs_update_new(self, db: NoteDB) -> None:
        assert db.note_needs_update("nueva", "cualquier contenido") is True

    def test_note_needs_update_same(self, db: NoteDB) -> None:
        db.upsert_note("n1", "T", [], 5, "contenido fijo", "/n1.md")
        assert db.note_needs_update("n1", "contenido fijo") is False

    def test_note_needs_update_changed(self, db: NoteDB) -> None:
        db.upsert_note("n1", "T", [], 5, "versión 1", "/n1.md")
        assert db.note_needs_update("n1", "versión 2") is True

    def test_get_content_hash(self, db: NoteDB) -> None:
        db.upsert_note("n1", "T", [], 5, "hola mundo", "/n1.md")
        h = db.get_content_hash("n1")
        assert h is not None
        assert len(h) == 64  # SHA-256 hex

    def test_get_content_hash_nonexistent(self, db: NoteDB) -> None:
        assert db.get_content_hash("no_existe") is None


# ===========================================================================
# Tests de Chunks
# ===========================================================================

class TestChunks:
    def test_upsert_and_get(self, db: NoteDB) -> None:
        db.upsert_note("parent", "Parent", [], 5, "...", "/p.md")
        db.upsert_chunk(
            "parent::sec1", "parent", "Sección 1", "Contenido sección 1.",
            ["tag1"], 7, 0, "structural",
        )
        chunks = db.get_chunks_for_note("parent")
        assert len(chunks) == 1
        c = chunks[0]
        assert c["id"] == "parent::sec1"
        assert c["note_id"] == "parent"
        assert c["title"] == "Sección 1"
        assert c["tags"] == ["tag1"]
        assert c["chunking_strategy"] == "structural"
        assert c["token_estimate"] > 0

    def test_get_all_chunks(self, populated_db: NoteDB) -> None:
        all_chunks = populated_db.get_all_chunks()
        assert len(all_chunks) == 5  # 3 de nota_a + 2 de nota_b

    def test_chunks_ordered(self, populated_db: NoteDB) -> None:
        chunks = populated_db.get_chunks_for_note("nota_a")
        orders = [c["order"] for c in chunks]
        assert orders == [0, 1, 2]

    def test_delete_chunks_for_note(self, populated_db: NoteDB) -> None:
        deleted = populated_db.delete_chunks_for_note("nota_a")
        assert deleted == 3
        assert populated_db.get_chunks_for_note("nota_a") == []
        # nota_b intacta
        assert len(populated_db.get_chunks_for_note("nota_b")) == 2

    def test_rechunk_note(self, populated_db: NoteDB) -> None:
        deleted = populated_db.rechunk_note("nota_a")
        assert deleted == 3

    def test_upsert_updates_chunk(self, db: NoteDB) -> None:
        db.upsert_note("p", "P", [], 5, "...", "/p.md")
        db.upsert_chunk("p::c1", "p", "V1", "original", [], 5, 0)
        db.upsert_chunk("p::c1", "p", "V2", "actualizado", [], 9, 0)
        chunks = db.get_chunks_for_note("p")
        assert len(chunks) == 1
        assert chunks[0]["title"] == "V2"
        assert chunks[0]["content"] == "actualizado"


# ===========================================================================
# Tests de Embeddings
# ===========================================================================

@skip_no_numpy
class TestEmbeddings:
    def test_save_and_load(self, populated_db: NoteDB) -> None:
        vec = np.random.randn(384).astype(np.float32)
        populated_db.save_embedding("nota_a::intro", vec)
        loaded = populated_db.load_embedding("nota_a::intro")
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, vec)

    def test_load_nonexistent(self, populated_db: NoteDB) -> None:
        assert populated_db.load_embedding("fantasma") is None

    def test_has_embedding(self, populated_db: NoteDB) -> None:
        assert populated_db.has_embedding("nota_a::intro") is False
        vec = np.zeros(384, dtype=np.float32)
        populated_db.save_embedding("nota_a::intro", vec)
        assert populated_db.has_embedding("nota_a::intro") is True

    def test_chunks_without_embeddings(self, populated_db: NoteDB) -> None:
        # Inicialmente ningún chunk tiene embedding
        without = populated_db.chunks_without_embeddings()
        assert len(without) == 5

        # Agregar embedding a uno
        vec = np.zeros(384, dtype=np.float32)
        populated_db.save_embedding("nota_a::intro", vec)
        without = populated_db.chunks_without_embeddings()
        assert len(without) == 4
        assert "nota_a::intro" not in without

    def test_load_all_embeddings(self, populated_db: NoteDB) -> None:
        v1 = np.ones(384, dtype=np.float32) * 0.1
        v2 = np.ones(384, dtype=np.float32) * 0.2
        populated_db.save_embedding("nota_a::intro", v1)
        populated_db.save_embedding("nota_b::modelo", v2)

        all_emb = populated_db.load_all_embeddings()
        assert len(all_emb) == 2
        assert "nota_a::intro" in all_emb
        assert "nota_b::modelo" in all_emb
        np.testing.assert_array_almost_equal(all_emb["nota_a::intro"], v1)

    def test_upsert_embedding(self, populated_db: NoteDB) -> None:
        v1 = np.ones(384, dtype=np.float32)
        v2 = np.ones(384, dtype=np.float32) * 2.0
        populated_db.save_embedding("nota_a::intro", v1)
        populated_db.save_embedding("nota_a::intro", v2)  # Sobreescribir
        loaded = populated_db.load_embedding("nota_a::intro")
        np.testing.assert_array_almost_equal(loaded, v2)

    def test_delete_embedding(self, populated_db: NoteDB) -> None:
        vec = np.zeros(384, dtype=np.float32)
        populated_db.save_embedding("nota_a::intro", vec)
        assert populated_db.delete_embedding("nota_a::intro") is True
        assert populated_db.load_embedding("nota_a::intro") is None
        assert populated_db.delete_embedding("nota_a::intro") is False


# ===========================================================================
# Tests de Cross-Links
# ===========================================================================

class TestCrossLinks:
    def test_save_and_get(self, populated_db: NoteDB) -> None:
        populated_db.save_link(
            "nota_a::intro", "nota_a::metricas", 0.85,
            "semantic_recurrence", "autocorrelation",
        )
        links = populated_db.get_links_for_chunk("nota_a::intro")
        assert len(links) == 1
        assert links[0]["target_chunk_id"] == "nota_a::metricas"
        assert links[0]["similarity"] == pytest.approx(0.85)

    def test_bidirectional_get(self, populated_db: NoteDB) -> None:
        populated_db.save_link(
            "nota_a::intro", "nota_b::modelo", 0.7,
        )
        # Desde source
        links = populated_db.get_links_for_chunk("nota_a::intro")
        assert len(links) == 1
        # Desde target
        links = populated_db.get_links_for_chunk("nota_b::modelo")
        assert len(links) == 1

    def test_outgoing_links(self, populated_db: NoteDB) -> None:
        populated_db.save_link("nota_a::intro", "nota_a::metricas", 0.8)
        populated_db.save_link("nota_a::metricas", "nota_a::intro", 0.6)

        outgoing = populated_db.get_outgoing_links("nota_a::intro")
        assert len(outgoing) == 1
        assert outgoing[0]["target_chunk_id"] == "nota_a::metricas"

    def test_save_links_batch(self, populated_db: NoteDB) -> None:
        links = [
            ("nota_a::intro", "nota_a::metricas", 0.8, "semantic_recurrence", "autocorrelation"),
            ("nota_a::intro", "nota_b::modelo", 0.7, "semantic_recurrence", "autocorrelation"),
            ("nota_a::rotaciones", "nota_b::cosine", 0.6, "semantic_recurrence", "autocorrelation"),
        ]
        count = populated_db.save_links_batch(links)
        assert count == 3

        all_links = populated_db.get_links_for_chunk("nota_a::intro")
        assert len(all_links) == 2

    def test_upsert_link(self, populated_db: NoteDB) -> None:
        populated_db.save_link("nota_a::intro", "nota_a::metricas", 0.5)
        populated_db.save_link("nota_a::intro", "nota_a::metricas", 0.9)
        links = populated_db.get_links_for_chunk("nota_a::intro")
        assert len(links) == 1
        assert links[0]["similarity"] == pytest.approx(0.9)

    def test_get_linked_chunks(self, populated_db: NoteDB) -> None:
        populated_db.save_link("nota_a::intro", "nota_b::modelo", 0.75)
        linked = populated_db.get_linked_chunks("nota_a::intro")
        assert len(linked) == 1
        assert linked[0]["id"] == "nota_b::modelo"
        assert linked[0]["similarity"] == pytest.approx(0.75)

    def test_delete_links_for_note(self, populated_db: NoteDB) -> None:
        populated_db.save_link("nota_a::intro", "nota_b::modelo", 0.7)
        populated_db.save_link("nota_a::rotaciones", "nota_b::cosine", 0.6)
        deleted = populated_db.delete_links_for_note("nota_a")
        assert deleted == 2


# ===========================================================================
# Tests de CASCADE
# ===========================================================================

class TestCascade:
    """Verifica que CASCADE DELETE funciona correctamente."""

    @skip_no_numpy
    def test_delete_note_cascades_chunks_and_embeddings(
        self, populated_db: NoteDB
    ) -> None:
        # Agregar embeddings
        vec = np.zeros(384, dtype=np.float32)
        populated_db.save_embedding("nota_a::intro", vec)
        populated_db.save_embedding("nota_a::rotaciones", vec)

        # Agregar links
        populated_db.save_link("nota_a::intro", "nota_a::metricas", 0.8)

        # Verificar estado antes
        assert len(populated_db.get_chunks_for_note("nota_a")) == 3
        assert populated_db.has_embedding("nota_a::intro") is True
        stats_before = populated_db.stats()

        # Borrar nota_a
        populated_db.delete_note("nota_a")

        # Verificar cascade
        assert populated_db.get_chunks_for_note("nota_a") == []
        assert populated_db.has_embedding("nota_a::intro") is False
        assert populated_db.has_embedding("nota_a::rotaciones") is False
        assert populated_db.get_links_for_chunk("nota_a::intro") == []

        # nota_b intacta
        assert len(populated_db.get_chunks_for_note("nota_b")) == 2

    def test_delete_chunk_cascades_links(self, populated_db: NoteDB) -> None:
        populated_db.save_link("nota_a::intro", "nota_b::modelo", 0.7)
        # Borrar el chunk source
        populated_db._conn.execute(
            "DELETE FROM chunks WHERE id = ?", ("nota_a::intro",)
        )
        populated_db._conn.commit()
        # El link debería desaparecer por CASCADE
        links = populated_db.get_links_for_chunk("nota_b::modelo")
        assert len(links) == 0


# ===========================================================================
# Tests de Utilidades
# ===========================================================================

class TestUtilities:
    def test_stats_empty(self, db: NoteDB) -> None:
        stats = db.stats()
        assert stats == {
            "notes": 0, "chunks": 0, "embeddings": 0, "chunk_links": 0
        }

    def test_stats_populated(self, populated_db: NoteDB) -> None:
        stats = populated_db.stats()
        assert stats["notes"] == 2
        assert stats["chunks"] == 5
        assert stats["embeddings"] == 0
        assert stats["chunk_links"] == 0

    def test_clear(self, populated_db: NoteDB) -> None:
        populated_db.clear()
        stats = populated_db.stats()
        assert all(v == 0 for v in stats.values())

    def test_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ctx.db"
        with NoteDB(db_path) as db:
            db.upsert_note("x", "X", [], 5, "...", "/x.md")
            note = db.get_note("x")
            assert note is not None
        # Después de salir, la conexión debería estar cerrada

    def test_vacuum(self, populated_db: NoteDB) -> None:
        populated_db.clear()
        populated_db.vacuum()  # No debería lanzar excepción


# ===========================================================================
# Tests de Búsqueda por Tags
# ===========================================================================

class TestSearchByTags:
    def test_search_single_tag(self, populated_db: NoteDB) -> None:
        results = populated_db.search_by_tags({"splay"}, top_k=10)
        assert len(results) >= 1
        ids = {r["id"] for r in results}
        assert "nota_a::intro" in ids

    def test_search_empty_tags(self, populated_db: NoteDB) -> None:
        results = populated_db.search_by_tags(set(), top_k=10)
        assert results == []

    def test_search_respects_top_k(self, populated_db: NoteDB) -> None:
        results = populated_db.search_by_tags({"cache", "embedding"}, top_k=2)
        assert len(results) <= 2
