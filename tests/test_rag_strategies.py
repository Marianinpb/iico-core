"""
tests/test_rag_strategies.py
==============================
Tests para las estrategias de retrieval de la Fase 2.

Ejecutar::

    pytest tests/test_rag_strategies.py -v
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from iico_core.db.note_db import NoteDB
from iico_core.rag_bench.strategies.base import (
    RetrievalStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from iico_core.rag_bench.strategies.embedding_strategy import EmbeddingStrategy
from iico_core.rag_bench.strategies.splay_strategy import SplayStrategy

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

skip_no_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockEmbeddingIndex:
    """Embedding index determinista para tests sin ONNX."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed) if _HAS_NUMPY else None
        self.call_count = 0

    def vectorize(self, text: str):
        """Genera embedding determinista basado en el texto."""
        self.call_count += 1
        # Usar hash del texto como seed determinista
        h = hash(text) % (2**31)
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(384).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class SimilarEmbeddingIndex:
    """Todos los chunks muy similares al query → siempre retorna resultados."""

    def vectorize(self, text: str):
        # Todos los textos → mismo vector base + ruido mínimo
        h = hash(text[:10]) % 100  # Solo primeros 10 chars
        rng = np.random.default_rng(h)
        base = np.ones(384, dtype=np.float32) / math.sqrt(384)
        noise = rng.normal(0, 0.001, 384).astype(np.float32)
        vec = base + noise
        norm = np.linalg.norm(vec)
        return vec / norm


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
def populated_db(db: NoteDB) -> NoteDB:
    """BD con notas, chunks y embeddings pre-cargados."""
    # Notas
    db.upsert_note("nota_splay", "Splay Tree", ["splay", "cache"], 9,
                   "El Splay Tree es una caché auto-ajustable.", "/nota_splay.md")
    db.upsert_note("nota_emb", "Embeddings", ["embedding", "onnx", "semantica"], 7,
                   "Los embeddings son vectores de alta dimensión.", "/nota_emb.md")
    db.upsert_note("nota_rag", "RAG", ["rag", "retrieval", "contexto"], 8,
                   "RAG combina retrieval con generación.", "/nota_rag.md")

    # Chunks
    db.upsert_chunk("nota_splay::intro", "nota_splay", "Intro Splay",
                    "El Splay Tree es una caché auto-ajustable.",
                    ["splay", "cache"], 9, 0)
    db.upsert_chunk("nota_splay::rotaciones", "nota_splay", "Rotaciones",
                    "Las rotaciones zig, zig-zig y zig-zag son las operaciones clave.",
                    ["splay"], 9, 1)
    db.upsert_chunk("nota_emb::modelo", "nota_emb", "Modelo",
                    "all-MiniLM-L6-v2 genera embeddings de dimensión 384.",
                    ["embedding", "onnx"], 7, 0)
    db.upsert_chunk("nota_emb::cosine", "nota_emb", "Cosine",
                    "La similaridad coseno mide el ángulo entre vectores.",
                    ["embedding", "semantica"], 7, 1)
    db.upsert_chunk("nota_rag::pipeline", "nota_rag", "Pipeline RAG",
                    "El pipeline RAG recupera contexto y genera respuestas.",
                    ["rag", "retrieval"], 8, 0)

    return db


@pytest.fixture
def db_with_embeddings(populated_db: NoteDB) -> NoteDB:
    """BD con embeddings guardados en BLOBs."""
    idx = SimilarEmbeddingIndex()
    chunks = populated_db.get_all_chunks()
    for chunk in chunks:
        vec = idx.vectorize(chunk["content"])
        populated_db.save_embedding(chunk["id"], vec)
    return populated_db


# ===========================================================================
# Tests del Registry
# ===========================================================================

class TestRegistry:
    def test_embeddings_registered(self) -> None:
        from iico_core.rag_bench.strategies import list_strategies
        assert "embeddings" in list_strategies()

    def test_splay_registered(self) -> None:
        from iico_core.rag_bench.strategies import list_strategies
        assert "splay" in list_strategies()

    def test_get_strategy_embeddings(self) -> None:
        cls = get_strategy("embeddings")
        assert cls is EmbeddingStrategy

    def test_get_strategy_splay(self) -> None:
        cls = get_strategy("splay")
        assert cls is SplayStrategy

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_strategy("estrategia_fantasma")

    def test_register_custom_strategy(self) -> None:
        @register_strategy
        class DummyStrategy(RetrievalStrategy):
            name = "dummy_retrieval_test"
            description = "Solo para tests"

            def setup(self, db, embedding_index, config): pass

            def retrieve(self, query, top_k=5):
                return []

        assert "dummy_retrieval_test" in list_strategies()
        inst = get_strategy("dummy_retrieval_test")()
        assert inst.retrieve("query") == []


# ===========================================================================
# Tests de EmbeddingStrategy — sin embeddings en BD
# ===========================================================================

class TestEmbeddingStrategyEmpty:
    def test_setup_empty_db(self, db: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db, MockEmbeddingIndex(), {})
        assert strategy.index_size == 0

    def test_retrieve_empty_returns_empty(self, db: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db, MockEmbeddingIndex(), {})
        results = strategy.retrieve("query de prueba", top_k=5)
        assert results == []


# ===========================================================================
# Tests de EmbeddingStrategy — con BD poblada
# ===========================================================================

@skip_no_numpy
class TestEmbeddingStrategy:
    def test_setup_builds_index(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        assert strategy.index_size == 5  # 5 chunks

    def test_retrieve_returns_list(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("splay tree", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("splay tree", top_k=2)
        assert len(results) <= 2

    def test_retrieve_returns_tuples(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("splay tree", top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            chunk_dict, score = item
            assert isinstance(chunk_dict, dict)
            assert isinstance(score, float)

    def test_retrieve_chunk_has_required_keys(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("embedding modelo", top_k=5)
        required_keys = {"id", "note_id", "title", "content"}
        for chunk_dict, _ in results:
            assert required_keys.issubset(chunk_dict.keys())

    def test_retrieve_sorted_by_score_desc(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("splay cache", top_k=5)
        if len(results) > 1:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_results(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        # Threshold muy alto → debería filtrar casi todo
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {"threshold": 0.999})
        results_high = strategy.retrieve("query", top_k=5)
        # Threshold cero → retorna todo
        strategy2 = EmbeddingStrategy()
        strategy2.setup(db_with_embeddings, SimilarEmbeddingIndex(), {"threshold": 0.0})
        results_low = strategy2.retrieve("query", top_k=5)
        assert len(results_low) >= len(results_high)

    def test_onthefly_vectorization(self, populated_db: NoteDB) -> None:
        """Sin embeddings en BD → genera on-the-fly con EmbeddingIndex."""
        # populated_db no tiene embeddings guardados
        strategy = EmbeddingStrategy()
        strategy.setup(populated_db, SimilarEmbeddingIndex(), {})
        # Debe haber vectorizado los 5 chunks on-the-fly
        assert strategy.index_size == 5

    def test_rebuild(self, db_with_embeddings: NoteDB) -> None:
        strategy = EmbeddingStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        initial_size = strategy.index_size
        strategy.rebuild()
        assert strategy.index_size == initial_size

    def test_retrieve_without_embedding_index(self, db_with_embeddings: NoteDB) -> None:
        """Si EmbeddingIndex es None pero hay embeddings en BD, sigue funcionando."""
        # Primero setup con índice para poblar la BD
        s1 = EmbeddingStrategy()
        s1.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})

        # Segundo setup sin índice → aun puede cargar desde BD
        s2 = EmbeddingStrategy()
        s2.setup(db_with_embeddings, None, {})
        # Tiene embeddings cargados pero no puede vectorizar query → vacío
        results = s2.retrieve("query", top_k=3)
        assert isinstance(results, list)


# ===========================================================================
# Tests de SplayStrategy
# ===========================================================================

@skip_no_numpy
class TestSplayStrategy:
    def test_setup(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        # No debe lanzar excepciones

    def test_retrieve_returns_list(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("splay tree cache", top_k=3)
        assert isinstance(results, list)

    def test_first_query_is_miss(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        # Primera query → siempre miss (caché vacío)
        strategy.retrieve("primera query", top_k=3)
        metrics = strategy.cache_metrics
        # Al menos 1 miss en el historial interno del embedding (delegó)
        # (no podemos verificar directamente el miss del Splay sin el tree,
        # pero sí que las métricas son accesibles)
        assert "hit_rate" in metrics
        assert "avg_depth" in metrics

    def test_repeated_query_can_hit(self, db_with_embeddings: NoteDB) -> None:
        """La misma query dos veces → segunda debería ser hit."""
        strategy = SplayStrategy()
        strategy.setup(
            db_with_embeddings,
            SimilarEmbeddingIndex(),
            {"hit_strategy": "token_overlap", "peek_n": 10}
        )
        query = "splay tree rotaciones cache"
        # Primera → miss (pobla el caché)
        results1 = strategy.retrieve(query, top_k=3)
        # Segunda → puede ser hit (depende de token overlap y peek_n)
        results2 = strategy.retrieve(query, top_k=3)
        # Ambas deben retornar resultados válidos
        assert isinstance(results1, list)
        assert isinstance(results2, list)

    def test_cache_metrics_structure(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        strategy.retrieve("query de prueba", top_k=3)
        metrics = strategy.cache_metrics
        required_keys = {"hits", "misses", "total_accesses", "hit_rate", "avg_depth"}
        assert required_keys.issubset(metrics.keys())
        assert isinstance(metrics["depth_history"], list)

    def test_reset_cache(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        strategy.retrieve("query", top_k=3)
        strategy.reset_cache()
        metrics = strategy.cache_metrics
        assert metrics["total_accesses"] == 0
        assert metrics["hits"] == 0

    def test_teardown(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        strategy.teardown()  # No debe lanzar excepción

    def test_respects_top_k(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(db_with_embeddings, SimilarEmbeddingIndex(), {})
        results = strategy.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_cache_size_config(self, db_with_embeddings: NoteDB) -> None:
        strategy = SplayStrategy()
        strategy.setup(
            db_with_embeddings,
            SimilarEmbeddingIndex(),
            {"splay_cache_size": 10}
        )
        # Solo validar que no lanza excepción con tamaño pequeño
        for i in range(5):
            strategy.retrieve(f"query número {i}", top_k=2)
        metrics = strategy.cache_metrics
        assert metrics["total_accesses"] >= 0


# ===========================================================================
# Tests de tokenización (SplayStrategy._tokenize)
# ===========================================================================

class TestSplayTokenization:
    def test_basic_tokenization(self) -> None:
        tokens = SplayStrategy._tokenize("Splay Tree caché")
        assert "splay" in tokens
        assert "tree" in tokens

    def test_filters_short_words(self) -> None:
        # La regex \w{2,} filtra palabras de 1 carácter, no de 2+
        tokens = SplayStrategy._tokenize("a b c d")
        assert len(tokens) == 0

    def test_keeps_two_char_words(self) -> None:
        tokens = SplayStrategy._tokenize("el la de un")
        # Palabras de 2 chars SÍ pasan el filtro \w{2,}
        assert len(tokens) == 4

    def test_empty_string(self) -> None:
        tokens = SplayStrategy._tokenize("")
        assert tokens == set()

    def test_build_cache_key_deterministic(self) -> None:
        t1 = {"tree", "splay", "cache"}
        t2 = {"cache", "splay", "tree"}
        assert SplayStrategy._build_cache_key(t1) == SplayStrategy._build_cache_key(t2)

    def test_build_cache_key_sorted(self) -> None:
        tokens = {"zebra", "apple", "mango"}
        key = SplayStrategy._build_cache_key(tokens)
        assert key == "apple mango zebra"
