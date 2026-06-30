"""
tests/test_benchmark_runner.py
================================
Tests para la Fase 3: métricas, RagasBridge, Runner y ReportGenerator.

Ejecutar::

    pytest tests/test_benchmark_runner.py -v
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from iico_core.rag_bench.types import (
    AggregatedMetrics,
    BenchmarkConfig,
    BenchmarkRun,
    DatasetQuery,
    IRMetrics,
    PerformanceMetrics,
    QueryResult,
    RAGASMetrics,
)
from iico_core.rag_bench.metrics.ir_metrics import (
    compute_ir_metrics,
    _precision, _recall, _mrr, _ndcg,
)
from iico_core.rag_bench.metrics.performance import (
    compute_performance_metrics,
    aggregate_metrics,
    _count_context_tokens,
    _compute_e_tok,
)
from iico_core.rag_bench.ragas_bridge import RagasBridge
from iico_core.rag_bench.runner import BenchmarkRunner
from iico_core.rag_bench.reports import ReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(chunk_id: str, content: str = "contenido de prueba") -> dict:
    return {
        "id": chunk_id,
        "note_id": "nota_test",
        "title": chunk_id,
        "content": content,
        "tags": ["test"],
        "priority": 5,
        "order": 0,
        "chunking_strategy": "structural",
    }


def make_query_result(
    query_id: str = "q01",
    retrieved_ids: list[str] | None = None,
    relevant_ids: list[str] | None = None,
    latency_ms: float = 10.0,
    ragas_score: float = 0.0,
    cache_hit: bool = False,
) -> QueryResult:
    retrieved_ids = retrieved_ids or ["chunk_1", "chunk_2"]
    relevant_ids = relevant_ids or ["chunk_1"]
    retrieved_chunks = [(make_chunk(cid, "x" * 400), 0.9 - i * 0.1)
                        for i, cid in enumerate(retrieved_ids)]
    ir = compute_ir_metrics(retrieved_ids, relevant_ids)
    perf = compute_performance_metrics(retrieved_chunks, latency_ms, ragas_score, cache_hit)
    return QueryResult(
        query_id=query_id,
        query=f"Query {query_id}",
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_ids,
        ir=ir,
        perf=perf,
    )


# ===========================================================================
# Tests de IRMetrics
# ===========================================================================

class TestIRMetrics:
    # ── Precision ───────────────────────────────────────────────────────
    def test_precision_all_relevant(self) -> None:
        assert _precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0

    def test_precision_none_relevant(self) -> None:
        assert _precision(["a", "b", "c"], {"x", "y"}) == 0.0

    def test_precision_half(self) -> None:
        assert _precision(["a", "b"], {"a", "x"}) == 0.5

    def test_precision_empty_retrieved(self) -> None:
        assert _precision([], {"a"}) == 0.0

    # ── Recall ──────────────────────────────────────────────────────────
    def test_recall_perfect(self) -> None:
        assert _recall(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_recall_zero(self) -> None:
        assert _recall(["a", "b"], {"x", "y"}, 2) == 0.0

    def test_recall_partial(self) -> None:
        result = _recall(["a", "b", "c"], {"a", "b", "d", "e"}, 4)
        assert result == 0.5

    def test_recall_no_relevant_in_gt(self) -> None:
        assert _recall(["a"], set(), 0) == 0.0

    # ── MRR ─────────────────────────────────────────────────────────────
    def test_mrr_first_position(self) -> None:
        assert _mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_mrr_second_position(self) -> None:
        assert abs(_mrr(["a", "b", "c"], {"b"}) - 0.5) < 1e-6

    def test_mrr_third_position(self) -> None:
        assert abs(_mrr(["a", "b", "c"], {"c"}) - 1/3) < 1e-6

    def test_mrr_no_relevant(self) -> None:
        assert _mrr(["a", "b", "c"], {"x"}) == 0.0

    def test_mrr_empty(self) -> None:
        assert _mrr([], {"a"}) == 0.0

    # ── nDCG ────────────────────────────────────────────────────────────
    def test_ndcg_perfect_order(self) -> None:
        # Relevante en primera posición → nDCG = 1
        ndcg = _ndcg(["a", "b", "c"], {"a"})
        assert ndcg == 1.0

    def test_ndcg_no_relevant(self) -> None:
        assert _ndcg(["a", "b"], {"x"}) == 0.0

    def test_ndcg_empty_retrieved(self) -> None:
        assert _ndcg([], {"a"}) == 0.0

    def test_ndcg_suboptimal_order(self) -> None:
        # Relevante en segunda posición → nDCG < 1
        ndcg = _ndcg(["a", "b"], {"b"})
        assert 0.0 < ndcg < 1.0

    # ── compute_ir_metrics ───────────────────────────────────────────────
    def test_compute_ir_metrics_perfect(self) -> None:
        m = compute_ir_metrics(["c1", "c2"], ["c1", "c2"])
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0
        assert m.mrr == 1.0
        assert m.ndcg == 1.0

    def test_compute_ir_metrics_zero(self) -> None:
        m = compute_ir_metrics(["c1", "c2"], ["x", "y"])
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0
        assert m.mrr == 0.0
        assert m.ndcg == 0.0

    def test_compute_ir_metrics_truncates_at_k(self) -> None:
        m = compute_ir_metrics(["c1", "c2", "c3"], ["c3"], k=2)
        # c3 NO está en los primeros 2 → MRR = 0
        assert m.mrr == 0.0
        assert m.num_retrieved == 2

    def test_compute_ir_metrics_counts(self) -> None:
        m = compute_ir_metrics(["c1", "c2", "c3"], ["c1", "c4"])
        assert m.num_retrieved == 3
        assert m.num_relevant == 2
        assert m.num_relevant_retrieved == 1


# ===========================================================================
# Tests de PerformanceMetrics
# ===========================================================================

class TestPerformanceMetrics:
    def test_token_count_estimate(self) -> None:
        chunks = [
            (make_chunk("c1", "a" * 400), 0.9),
            (make_chunk("c2", "b" * 400), 0.8),
        ]
        tokens = _count_context_tokens(chunks)
        assert tokens == 200  # 800 chars / 4

    def test_token_count_empty(self) -> None:
        assert _count_context_tokens([]) == 0

    def test_e_tok_normal(self) -> None:
        # 500 tokens, RAGAS=0.8 → 500/0.81 ≈ 617
        e = _compute_e_tok(500, 0.8)
        assert abs(e - 500 / 0.81) < 0.1

    def test_e_tok_zero_tokens(self) -> None:
        assert _compute_e_tok(0, 0.8) == 0.0

    def test_e_tok_zero_ragas_no_div_zero(self) -> None:
        # ragas=0 → usa ε=0.01 → 500/0.01 = 50000
        e = _compute_e_tok(500, 0.0)
        assert e == 500 / 0.01

    def test_compute_performance_metrics(self) -> None:
        chunks = [(make_chunk("c1", "x" * 400), 0.9)]
        perf = compute_performance_metrics(chunks, 15.5, ragas_score=0.7)
        assert perf.retrieval_latency_ms == 15.5
        assert perf.total_context_tokens == 100
        assert perf.ragas_score == 0.7
        assert perf.e_tok > 0
        assert perf.cache_hit is False

    def test_cache_hit_flag(self) -> None:
        perf = compute_performance_metrics([], 5.0, cache_hit=True)
        assert perf.cache_hit is True


# ===========================================================================
# Tests de aggregate_metrics
# ===========================================================================

class TestAggregateMetrics:
    def test_empty_returns_zeros(self) -> None:
        agg = aggregate_metrics([])
        assert agg.num_queries == 0
        assert agg.precision_at_k == 0.0

    def test_single_query(self) -> None:
        qr = make_query_result("q01", ["c1"], ["c1"], latency_ms=20.0)
        agg = aggregate_metrics([qr])
        assert agg.num_queries == 1
        assert agg.precision_at_k == 1.0
        assert agg.recall_at_k == 1.0
        assert agg.avg_latency_ms == 20.0

    def test_multiple_queries_avg(self) -> None:
        qr1 = make_query_result("q01", ["c1"], ["c1"], latency_ms=10.0)
        qr2 = make_query_result("q02", ["c2"], ["c1"], latency_ms=30.0)
        agg = aggregate_metrics([qr1, qr2])
        assert agg.num_queries == 2
        assert agg.avg_latency_ms == 20.0
        # q01 perfecta, q02 con 0 hits → avg precision = 0.5
        assert abs(agg.precision_at_k - 0.5) < 1e-6

    def test_splay_hit_rate(self) -> None:
        qr1 = make_query_result("q01", cache_hit=True)
        qr2 = make_query_result("q02", cache_hit=False)
        agg = aggregate_metrics([qr1, qr2])
        assert agg.splay_hit_rate == 0.5

    def test_ragas_cache_hit_rate(self) -> None:
        qr1 = make_query_result("q01")
        qr1.ragas = RAGASMetrics(from_cache=True)
        qr1.perf.ragas_score = 0.8

        qr2 = make_query_result("q02")
        qr2.ragas = RAGASMetrics(from_cache=False)
        qr2.perf.ragas_score = 0.7

        agg = aggregate_metrics([qr1, qr2])
        assert agg.ragas_cache_hit_rate == 0.5


# ===========================================================================
# Tests de RAGASMetrics
# ===========================================================================

class TestRAGASMetrics:
    def test_aggregate_score(self) -> None:
        m = RAGASMetrics(faithfulness=0.8, answer_relevancy=0.6)
        assert abs(m.aggregate - 0.7) < 1e-6

    def test_aggregate_zero_if_error(self) -> None:
        m = RAGASMetrics(error="no_api_key")
        assert m.aggregate == 0.0


# ===========================================================================
# Tests de RagasBridge (sin API real)
# ===========================================================================

class TestRagasBridge:
    def test_no_api_key_returns_error(self, tmp_path: Path) -> None:
        bridge = RagasBridge(api_key="", cache_path=tmp_path / "cache.json")
        result = bridge.evaluate("query", ["contexto"])
        assert result.error == "no_api_key"

    def test_empty_contexts_returns_error(self, tmp_path: Path) -> None:
        bridge = RagasBridge(api_key="sk-fake", cache_path=tmp_path / "cache.json")
        result = bridge.evaluate("query", [])
        assert result.error == "no_contexts"

    def test_cache_miss_then_hit(self, tmp_path: Path, monkeypatch) -> None:
        """Con monkeypatch del _call_deepseek → testear caché sin API real."""
        call_count = {"n": 0}

        def mock_call(self, prompt: str) -> str:
            call_count["n"] += 1
            return "0.75"

        monkeypatch.setattr(RagasBridge, "_call_deepseek", mock_call)

        bridge = RagasBridge(api_key="sk-fake", cache_path=tmp_path / "cache.json")
        contexts = ["contexto A", "contexto B"]

        # Primera llamada → miss → llama 4 veces (4 métricas)
        result1 = bridge.evaluate("¿qué es el splay tree?", contexts)
        assert result1.from_cache is False
        assert call_count["n"] == 4

        # Segunda llamada con mismos contexts → hit
        result2 = bridge.evaluate("¿qué es el splay tree?", contexts)
        assert result2.from_cache is True
        assert call_count["n"] == 4  # No se hicieron más llamadas

    def test_cache_hash_order_independent(self, tmp_path: Path, monkeypatch) -> None:
        """El orden de los contextos no afecta el hash."""
        monkeypatch.setattr(RagasBridge, "_call_deepseek", lambda self, p: "0.8")

        bridge = RagasBridge(api_key="sk-fake", cache_path=tmp_path / "cache.json")
        query = "query de prueba"
        ctx_a = ["contexto 1", "contexto 2"]
        ctx_b = ["contexto 2", "contexto 1"]

        bridge.evaluate(query, ctx_a)
        result = bridge.evaluate(query, ctx_b)
        assert result.from_cache is True  # Mismo hash aunque orden diferente

    def test_cache_persisted_to_disk(self, tmp_path: Path, monkeypatch) -> None:
        """El caché se guarda en disco y se carga en nuevas instancias."""
        monkeypatch.setattr(RagasBridge, "_call_deepseek", lambda self, p: "0.9")
        cache_path = tmp_path / "cache.json"

        bridge1 = RagasBridge(api_key="sk-fake", cache_path=cache_path)
        bridge1.evaluate("query", ["ctx"])

        # Nueva instancia lee el caché del disco
        bridge2 = RagasBridge(api_key="sk-fake", cache_path=cache_path)
        assert bridge2.cache_size == 1

    def test_cache_size(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(RagasBridge, "_call_deepseek", lambda self, p: "0.7")
        bridge = RagasBridge(api_key="sk-fake", cache_path=tmp_path / "c.json")

        bridge.evaluate("query 1", ["ctx 1"])
        bridge.evaluate("query 2", ["ctx 2"])
        assert bridge.cache_size == 2

    def test_clear_cache(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(RagasBridge, "_call_deepseek", lambda self, p: "0.8")
        cache_path = tmp_path / "cache.json"
        bridge = RagasBridge(api_key="sk-fake", cache_path=cache_path)
        bridge.evaluate("q", ["ctx"])
        assert bridge.cache_size == 1
        bridge.clear_cache()
        assert bridge.cache_size == 0
        assert not cache_path.exists()

    def test_parse_score_valid(self) -> None:
        assert RagasBridge._parse_score("0.75") == 0.75
        assert RagasBridge._parse_score("Score: 0.85") == 0.85

    def test_parse_score_clamped(self) -> None:
        assert RagasBridge._parse_score("1.5") == 1.0
        assert RagasBridge._parse_score("-0.5") == 0.0

    def test_parse_score_no_number(self) -> None:
        assert RagasBridge._parse_score("I cannot answer") == 0.0

    def test_hash_deterministic(self) -> None:
        h1 = RagasBridge._hash_contexts("q", ["a", "b"])
        h2 = RagasBridge._hash_contexts("q", ["a", "b"])
        assert h1 == h2

    def test_hash_different_queries(self) -> None:
        h1 = RagasBridge._hash_contexts("q1", ["ctx"])
        h2 = RagasBridge._hash_contexts("q2", ["ctx"])
        assert h1 != h2


# ===========================================================================
# Tests de BenchmarkRunner (sin EmbeddingIndex real)
# ===========================================================================

class TestBenchmarkRunner:
    """Tests del runner con una estrategia mock y dataset YAML en disco."""

    @pytest.fixture
    def notes_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "notes"
        d.mkdir()
        # Crear una nota de prueba
        note = d / "splay.md"
        note.write_text(
            "---\nid: nota-splay\ntags: [splay]\npriority: 8\n---\n"
            "# Splay Tree\n\n## Introduccion\n\nEl Splay Tree es una estructura de datos.",
            encoding="utf-8",
        )
        return d

    @pytest.fixture
    def dataset_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "dataset.yaml"
        p.write_text(
            "name: test\nqueries:\n"
            "  - id: q01\n"
            "    query: \"¿Qué es el Splay Tree?\"\n"
            "    relevant_chunk_ids: [\"nota-splay::introduccion\"]\n"
            "    expected_answer: \"Es una estructura de datos.\"\n",
            encoding="utf-8",
        )
        return p

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "bench.db"

    def test_run_without_embedding_index(
        self, db_path: Path, notes_dir: Path, dataset_path: Path
    ) -> None:
        """Ejecutar con estrategia 'embeddings' sin ONNX debe completar sin crash."""
        config = BenchmarkConfig(
            chunking_pipeline=[("structural", {})],
            retrieval_strategy="embeddings",
            dataset_path=str(dataset_path),
            notes_dir=str(notes_dir),
            enable_ragas=False,
        )
        with BenchmarkRunner(db_path=db_path, verbose=False) as runner:
            run = runner.run(config)

        assert isinstance(run, BenchmarkRun)
        assert run.aggregated.num_queries == 1
        # Sin embeddings reales, los resultados serán vacíos pero no debe crashear
        assert run.aggregated.precision_at_k >= 0.0

    def test_run_loads_dataset_queries(
        self, db_path: Path, notes_dir: Path, dataset_path: Path
    ) -> None:
        config = BenchmarkConfig(
            chunking_pipeline=[("structural", {})],
            retrieval_strategy="embeddings",
            dataset_path=str(dataset_path),
            notes_dir=str(notes_dir),
        )
        with BenchmarkRunner(db_path=db_path, verbose=False) as runner:
            run = runner.run(config)

        assert len(run.query_results) == 1
        assert run.query_results[0].query_id == "q01"

    def test_run_missing_dataset_returns_empty(
        self, db_path: Path, notes_dir: Path
    ) -> None:
        config = BenchmarkConfig(
            chunking_pipeline=[("structural", {})],
            retrieval_strategy="embeddings",
            dataset_path="archivo_que_no_existe.yaml",
            notes_dir=str(notes_dir),
        )
        with BenchmarkRunner(db_path=db_path, verbose=False) as runner:
            run = runner.run(config)

        assert len(run.query_results) == 0
        assert len(run.errors) > 0

    def test_run_id_auto_generated(self) -> None:
        config = BenchmarkConfig(
            chunking_pipeline=[("structural", {})],
            retrieval_strategy="embeddings",
        )
        assert "structural" in config.run_id
        assert "embeddings" in config.run_id

    def test_context_manager(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        with BenchmarkRunner(db_path=db_path, verbose=False) as runner:
            assert runner is not None


# ===========================================================================
# Tests de ReportGenerator
# ===========================================================================

class TestReportGenerator:
    @pytest.fixture
    def sample_run(self) -> BenchmarkRun:
        config = BenchmarkConfig(
            chunking_pipeline=[("structural", {})],
            retrieval_strategy="embeddings",
            run_id="test__structural__embeddings",
            enable_ragas=False,
        )
        qr1 = make_query_result("q01", ["c1", "c2"], ["c1"], latency_ms=15.0)
        qr2 = make_query_result("q02", ["c3"], ["c3"], latency_ms=8.0)
        agg = aggregate_metrics([qr1, qr2])

        return BenchmarkRun(
            config=config,
            query_results=[qr1, qr2],
            aggregated=agg,
        )

    def test_generate_creates_files(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        assert paths["markdown"].exists()
        assert paths["csv"].exists()
        assert paths["json"].exists()

    def test_markdown_contains_run_id(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        content = paths["markdown"].read_text(encoding="utf-8")
        assert sample_run.config.run_id in content

    def test_markdown_contains_metrics(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        content = paths["markdown"].read_text(encoding="utf-8")
        assert "Precision@K" in content
        assert "nDCG" in content

    def test_csv_has_header(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        with open(paths["csv"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "precision_at_k" in headers
        assert "latency_ms" in headers
        assert "context_tokens" in headers

    def test_csv_row_count(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        with open(paths["csv"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # 2 queries

    def test_json_summary_structure(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert data["run_id"] == sample_run.config.run_id
        assert "aggregated" in data
        assert "ndcg" in data["aggregated"]

    def test_compare_generates_file(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        comparison_path = gen.compare([sample_run, sample_run])
        assert comparison_path.exists()
        content = comparison_path.read_text(encoding="utf-8")
        assert "Comparación" in content
        assert "structural" in content

    def test_no_dsp_file_without_signals(self, tmp_path: Path, sample_run: BenchmarkRun) -> None:
        gen = ReportGenerator(tmp_path / "results")
        paths = gen.generate(sample_run)
        assert "dsp" not in paths  # No hay señales DSP en este run
