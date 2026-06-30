"""
iico_core/rag_bench/runner.py
===============================
Orquestador del benchmark: conecta todas las piezas y ejecuta los runs.

Flujo de un run::

    BenchmarkRunner.run(config)
         │
         ├── 1. Cargar dataset (YAML)
         ├── 2. Sincronizar test_notes con NoteDB (NoteWatcher.sync())
         ├── 3. Chunking con ChunkingPipeline
         ├── 4. Setup de la estrategia de retrieval
         └── 5. Por cada query del dataset:
                  ├── medir latencia de retrieval
                  ├── calcular IRMetrics vs ground truth
                  ├── calcular PerformanceMetrics (tokens, E_tok)
                  ├── (opcional) RagasBridge.evaluate()
                  └── acumular en QueryResult

    → BenchmarkRun con aggregated metrics
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from .chunking import ChunkingPipeline
from .metrics import aggregate_metrics, compute_performance_metrics
from .ragas_bridge import RagasBridge
from .strategies import get_strategy
from .types import (
    AggregatedMetrics,
    BenchmarkConfig,
    BenchmarkRun,
    DatasetQuery,
    PerformanceMetrics,
    QueryResult,
    RAGASMetrics,
)
from ..db.note_db import NoteDB
from ..db.watcher import NoteWatcher

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orquestador del benchmark RAG.

    Conecta NoteDB, NoteWatcher, ChunkingPipeline, RetrievalStrategy,
    métricas IR, E_tok y RagasBridge en un flujo de un solo método: ``run()``.

    Args:
        db_path: Ruta a la BD SQLite. Default: ``"bench_iico.db"``.
        embedding_index: Instancia de :class:`EmbeddingIndex` (opcional).
                         Si es None, las estrategias que requieran embeddings
                         usarán embeddings on-the-fly si los chunks ya están en BD.
        ragas_bridge: Instancia de :class:`RagasBridge` (opcional).
                      Si es None, RAGAS no se calcula aunque ``config.enable_ragas=True``.
        verbose: Si True, imprime progreso por consola.
    """

    def __init__(
        self,
        db_path: str | Path = "bench_iico.db",
        embedding_index: Any = None,
        ragas_bridge: RagasBridge | None = None,
        verbose: bool = True,
    ) -> None:
        self._db = NoteDB(db_path)
        self._embedding_index = embedding_index
        self._ragas_bridge = ragas_bridge
        self._verbose = verbose

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, config: BenchmarkConfig) -> BenchmarkRun:
        """Ejecuta el benchmark completo para una configuración.

        Args:
            config: Configuración del run.

        Returns:
            :class:`BenchmarkRun` con resultados individuales y agregados.
        """
        t_run_start = time.perf_counter()
        self._log(f"\n{'='*60}")
        self._log(f"  Benchmark run: {config.run_id}")
        self._log(f"{'='*60}")

        run = BenchmarkRun(config=config)
        chunking_latency_ms = 0.0

        # ── 1. Cargar dataset ──────────────────────────────────────────
        queries = self._load_dataset(config.dataset_path)
        if not queries:
            run.errors.append(f"Dataset vacío o no encontrado: {config.dataset_path}")
            return run

        self._log(f"  Dataset: {len(queries)} queries")

        # ── 2. Sincronizar notas con NoteDB ────────────────────────────
        if config.notes_dir:
            report = self._sync_notes(config)
            chunking_latency_ms = report.elapsed_ms

        # ── 2.5. Calentamiento de ONNX (Prevenir Cold Start) ────────────
        if self._embedding_index is not None:
            # Forzar la carga en memoria del modelo antes de medir latencia
            try:
                self._embedding_index.vectorize("warmup")
            except Exception:
                pass

        # ── 3. Setup de la estrategia de retrieval ─────────────────────
        try:
            retrieval_cfg = dict(config.retrieval_config) if config.retrieval_config else {}
            # Pasamos el nombre de la estrategia para que Splay/Embeddings aíslen su caché
            pipeline_name = ChunkingPipeline(config.chunking_pipeline).name
            retrieval_cfg["chunking_strategy_name"] = pipeline_name
            
            strategy_cls = get_strategy(config.retrieval_strategy)
            strategy = strategy_cls()
            strategy.setup(self._db, self._embedding_index, retrieval_cfg)
        except Exception as e:
            run.errors.append(f"Error inicializando estrategia: {e}")
            return run

        self._log(f"  Estrategia: {config.retrieval_strategy}")
        self._log(f"  Chunking: {' -> '.join(n for n, _ in config.chunking_pipeline)}")
        self._log(f"  Top-K: {config.top_k}")
        self._log("")

        # ── 4. Evaluar cada query ──────────────────────────────────────
        for i, dataset_query in enumerate(queries, start=1):
            self._log(f"  [{i:03d}/{len(queries)}] {dataset_query.query[:60]}")
            try:
                qr = self._evaluate_query(
                    dataset_query=dataset_query,
                    strategy=strategy,
                    config=config,
                )
                run.query_results.append(qr)
            except Exception as e:
                msg = f"Error en query '{dataset_query.id}': {e}"
                logger.error("[BenchmarkRunner] %s", msg)
                run.errors.append(msg)

        # ── 5. Agregar métricas ────────────────────────────────────────
        run.aggregated = aggregate_metrics(run.query_results)
        run.aggregated.chunking_latency_ms = chunking_latency_ms
        run.aggregated.total_elapsed_ms = (time.perf_counter() - t_run_start) * 1000

        # Splay hit rate desde las métricas del árbol
        if hasattr(strategy, "cache_metrics"):
            cm = strategy.cache_metrics
            run.aggregated.splay_hit_rate = cm.get("hit_rate", 0.0)

        strategy.teardown()

        self._log(f"\n  -- Resultados ------------------")
        self._log(f"  Avg Latency:    {run.aggregated.avg_latency_ms:.1f}ms")
        self._log(f"  Avg Tokens:     {run.aggregated.avg_context_tokens:.0f}")
        if config.enable_ragas:
            self._log(f"  Avg RAGAS:      {run.aggregated.avg_ragas_score:.4f}")
            self._log(f"  RAGAS P:        {run.aggregated.avg_ragas_context_precision:.4f}")
            self._log(f"  RAGAS R:        {run.aggregated.avg_ragas_context_recall:.4f}")
            self._log(f"  Avg E_tok:      {run.aggregated.avg_e_tok:.1f}")
            self._log(f"  RAGAS cache HR: {run.aggregated.ragas_cache_hit_rate:.2%}")
        self._log(f"  Total elapsed:  {run.aggregated.total_elapsed_ms:.0f}ms")
        if run.errors:
            self._log(f"  Errors:         {len(run.errors)}")

        return run

    def compare(self, configs: list[BenchmarkConfig]) -> list[BenchmarkRun]:
        """Ejecuta múltiples configs y retorna todos los runs.

        Útil para comparar directamente chunkers/estrategias.
        Los resultados se pueden pasar al :class:`ReportGenerator`.
        """
        return [self.run(config) for config in configs]

    def close(self) -> None:
        """Cierra la conexión a la BD."""
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Evaluación de una query
    # ------------------------------------------------------------------

    def _evaluate_query(
        self,
        dataset_query: DatasetQuery,
        strategy: Any,
        config: BenchmarkConfig,
    ) -> QueryResult:
        """Evalúa una sola query y retorna su :class:`QueryResult`."""

        # ── Retrieval con medición de latencia ─────────────────────────
        t0 = time.perf_counter()
        retrieved_chunks = strategy.retrieve(dataset_query.query, config.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Detectar cache_hit del Splay
        cache_hit = False
        if hasattr(strategy, "cache_metrics"):
            prev_hits = getattr(strategy, "_prev_hits", 0)
            curr_hits = strategy.cache_metrics.get("hits", 0)
            cache_hit = curr_hits > prev_hits
            strategy._prev_hits = curr_hits

        # ── RAGAS (opcional) ───────────────────────────────────────────
        ragas_metrics: RAGASMetrics | None = None
        ragas_score = 0.0

        if config.enable_ragas and self._ragas_bridge is not None and retrieved_chunks:
            contexts = [c["content"] for c, _ in retrieved_chunks]
            ragas_metrics = self._ragas_bridge.evaluate(
                query=dataset_query.query,
                contexts=contexts,
                expected_answer=dataset_query.expected_answer,
            )
            if ragas_metrics.error is None:
                ragas_score = ragas_metrics.aggregate

        # ── Métricas de performance ────────────────────────────────────
        perf = compute_performance_metrics(
            retrieved_chunks=retrieved_chunks,
            retrieval_latency_ms=latency_ms,
            ragas_score=ragas_score,
            cache_hit=cache_hit,
        )

        log_str = (
            f"         lat={latency_ms:.1f}ms  "
            f"tok={perf.total_context_tokens}"
        )
        if ragas_metrics:
            log_str += f"  RAGAS_P={ragas_metrics.context_precision:.2f}  RAGAS_R={ragas_metrics.context_recall:.2f}"
        
        self._log(log_str)

        return QueryResult(
            query_id=dataset_query.id,
            query=dataset_query.query,
            retrieved_chunks=retrieved_chunks,
            relevant_chunk_ids=dataset_query.relevant_chunk_ids,
            perf=perf,
            ragas=ragas_metrics,
        )

    # ------------------------------------------------------------------
    # Carga de dataset
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset(dataset_path: str) -> list[DatasetQuery]:
        """Carga queries desde un archivo YAML.

        Formato esperado::

            name: mi_dataset
            queries:
              - id: q01
                query: "¿Qué es el Splay Tree?"
                relevant_chunk_ids:
                  - "nota-splay::intro"
                expected_answer: "Es una caché auto-ajustable."
                tags: [splay, cache]

        Args:
            dataset_path: Ruta al archivo YAML.

        Returns:
            Lista de :class:`DatasetQuery`. Vacía si el archivo no existe.
        """
        path = Path(dataset_path)
        if not path.exists():
            logger.warning("[BenchmarkRunner] Dataset no encontrado: %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error("[BenchmarkRunner] Error cargando dataset: %s", e)
            return []

        queries = []
        for item in data.get("queries", []):
            queries.append(DatasetQuery(
                id=str(item.get("id", "")),
                query=str(item.get("query", "")),
                relevant_chunk_ids=list(item.get("relevant_chunk_ids", [])),
                expected_answer=str(item.get("expected_answer", "")),
                tags=list(item.get("tags", [])),
            ))
        return queries

    # ------------------------------------------------------------------
    # Sincronización de notas
    # ------------------------------------------------------------------

    def _sync_notes(self, config: BenchmarkConfig) -> Any:
        """Sincroniza el dataset de notas usando el ChunkingPipeline del config."""
        pipeline = ChunkingPipeline(config.chunking_pipeline)
        pipeline.setup(embedding_index=self._embedding_index)
        
        watcher = NoteWatcher(
            notes_dir=config.notes_dir,
            db=self._db,
            chunker=pipeline,
            embedding_index=self._embedding_index,
            chunking_strategy_name=config.run_id,
        )
        report = watcher.sync()
        self._log(
            f"  NoteWatcher: {report.new_notes} new, "
            f"{report.modified_notes} modified, "
            f"{report.unchanged_notes} unchanged, "
            f"{report.total_chunks} chunks"
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)
