"""
iico_core/rag_bench/types.py
==============================
Tipos de datos compartidos por todo el sistema de benchmark.

Jerarquía de resultados::

    BenchmarkConfig
         │
         ▼
    BenchmarkRun   (1 run = 1 config de chunking + 1 estrategia de retrieval)
         │
         ├── QueryResult  (1 por cada query del dataset)
         │       ├── IRMetrics
         │       ├── PerformanceMetrics
         │       └── RAGASMetrics  (opcional, si hay caché disponible)
         │
         └── AggregatedMetrics  (promedio/suma sobre todos los QueryResults)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuración de un run
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Define completamente un experimento del benchmark.

    Identifica de forma única la combinación de estrategias a evaluar.
    Dos configs con el mismo ``run_id`` son equivalentes.

    Attributes:
        run_id: Identificador único del run. Se genera automáticamente si
                no se provee (basado en chunking+strategy).
        chunking_pipeline: Lista de tuplas ``(nombre_chunker, config_dict)``
                           para el :class:`ChunkingPipeline`.
        retrieval_strategy: Nombre de la estrategia de retrieval registrada.
        retrieval_config: Parámetros de la estrategia de retrieval.
        top_k: Número máximo de chunks a recuperar por query.
        dataset_path: Ruta al archivo YAML con las queries del dataset.
        notes_dir: Carpeta con las notas ``.md`` a indexar.
        enable_ragas: Si True, llama a RAGAS para métricas de calidad del LLM.
        ragas_model: Modelo LLM para el ``RagasBridge``. Default: ``"deepseek-chat"``.
        description: Descripción opcional del run para los reportes.
    """
    chunking_pipeline: list[tuple[str, dict]]
    retrieval_strategy: str
    retrieval_config: dict = field(default_factory=dict)
    top_k: int = 5
    dataset_path: str = ""
    notes_dir: str = ""
    enable_ragas: bool = False
    ragas_model: str = "deepseek-chat"
    run_id: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            chunkers = "+".join(name for name, _ in self.chunking_pipeline)
            self.run_id = f"{chunkers}__{self.retrieval_strategy}"


# ---------------------------------------------------------------------------
# Entrada del dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetQuery:
    """Una query del dataset de evaluación.

    Attributes:
        id: Identificador único de la query dentro del dataset.
        query: Texto de la consulta en lenguaje natural.
        relevant_chunk_ids: IDs de los chunks que son ground truth (relevantes).
                            Usados para calcular Precision/Recall/MRR/nDCG.
        expected_answer: Respuesta esperada (para RAGAS faithfulness/answer_relevancy).
        tags: Etiquetas opcionales para filtrar subconjuntos del dataset.
    """
    id: str
    query: str
    relevant_chunk_ids: list[str]
    expected_answer: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Métricas de recuperación de información (IR)
# ---------------------------------------------------------------------------

@dataclass
class IRMetrics:
    """Métricas clásicas de Information Retrieval para una query.

    Calculadas comparando los chunks recuperados contra el ground truth.

    Attributes:
        precision_at_k: Fracción de chunks recuperados que son relevantes.
                        ``|retrieved ∩ relevant| / k``
        recall_at_k: Fracción de relevantes que fueron recuperados.
                     ``|retrieved ∩ relevant| / |relevant|``
        mrr: Mean Reciprocal Rank. Posición del primer chunk relevante.
             ``1/rank_of_first_relevant``; 0 si ninguno es relevante.
        ndcg: Normalized Discounted Cumulative Gain.
              Considera el orden de los resultados relevantes.
        num_retrieved: Número de chunks efectivamente recuperados.
        num_relevant: Número de chunks relevantes en el ground truth.
        num_relevant_retrieved: Intersección.
    """
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    num_retrieved: int = 0
    num_relevant: int = 0
    num_relevant_retrieved: int = 0


# ---------------------------------------------------------------------------
# Métricas de performance
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento para una query.

    Attributes:
        retrieval_latency_ms: Tiempo de recuperación (ms).
        total_context_tokens: Tokens totales en todos los chunks recuperados.
                              Estimación: len(content) // 4 por chunk.
        ragas_score: Score agregado de RAGAS (0.0 si no se calculó).
        e_tok: Métrica de eficiencia de tokens.
               ``E_tok = total_context_tokens / (ragas_score + ε)``.
               Menor E_tok = más eficiente (menos tokens, misma calidad).
        cache_hit: True si fue hit del Splay Tree.
    """
    retrieval_latency_ms: float = 0.0
    total_context_tokens: int = 0
    ragas_score: float = 0.0
    e_tok: float = 0.0
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# Métricas de RAGAS
# ---------------------------------------------------------------------------

@dataclass
class RAGASMetrics:
    """Métricas de calidad RAG calculadas con el framework RAGAS.

    Attributes:
        faithfulness: Qué tan fiel es la respuesta al contexto recuperado.
                      (0.0 = inventa, 1.0 = completamente basado en contexto)
        answer_relevancy: Qué tan relevante es la respuesta para la pregunta.
        context_precision: Precisión del contexto para la pregunta.
        context_recall: Recall del contexto para la pregunta.
        from_cache: Si True, este resultado vino del caché (no costó tokens).
        context_hash: Hash SHA-256 de los contextos usados (para deduplicar).
        error: Mensaje de error si RAGAS falló (None si exitoso).
    """
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    from_cache: bool = False
    context_hash: str = ""
    error: str | None = None

    @property
    def aggregate(self) -> float:
        """Score agregado: promedio de faithfulness y answer_relevancy.

        Estas dos métricas son las más directas para evaluar la calidad
        del contexto recuperado en el objetivo de la tesis.
        """
        return (self.faithfulness + self.answer_relevancy) / 2.0


# ---------------------------------------------------------------------------
# Resultado de una query individual
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Resultado completo de ejecutar una query en un run del benchmark.

    Attributes:
        query_id: ID de la query del dataset.
        query: Texto de la consulta.
        retrieved_chunks: Lista de ``(chunk_dict, score)`` recuperados.
        relevant_chunk_ids: Ground truth para métricas IR.
        ir: Métricas de Information Retrieval.
        perf: Métricas de performance.
        ragas: Métricas RAGAS (None si no se calcularon).
        dsp_metadata: Señales DSP del chunker para esta nota (si aplica).
    """
    query_id: str
    query: str
    retrieved_chunks: list[tuple[dict, float]]
    relevant_chunk_ids: list[str]
    perf: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    ragas: RAGASMetrics | None = None
    dsp_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_ids(self) -> list[str]:
        """IDs de los chunks recuperados, en orden de score."""
        return [c["id"] for c, _ in self.retrieved_chunks]


# ---------------------------------------------------------------------------
# Métricas agregadas del run
# ---------------------------------------------------------------------------

@dataclass
class AggregatedMetrics:
    """Promedios de todas las queries del run.

    Attributes:
        num_queries: Número de queries evaluadas.
        precision_at_k: Promedio de Precision@K.
        recall_at_k: Promedio de Recall@K.
        mrr: Promedio de MRR.
        ndcg: Promedio de nDCG.
        avg_latency_ms: Latencia promedio de retrieval.
        avg_context_tokens: Tokens promedio por query.
        avg_ragas_score: Score RAGAS promedio (0 si no se usó RAGAS).
        avg_e_tok: Eficiencia de tokens promedio.
        ragas_cache_hit_rate: Fracción de queries que usaron el caché RAGAS.
        splay_hit_rate: Hit rate del Splay Tree (0 si no se usó).
        total_elapsed_ms: Tiempo total del run.
    """
    num_queries: int = 0
    avg_latency_ms: float = 0.0
    chunking_latency_ms: float = 0.0
    avg_context_tokens: float = 0.0
    avg_ragas_score: float = 0.0
    avg_ragas_context_precision: float = 0.0
    avg_ragas_context_recall: float = 0.0
    avg_e_tok: float = 0.0
    ragas_cache_hit_rate: float = 0.0
    splay_hit_rate: float = 0.0
    total_elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Resultado completo de un run
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    """Resultado completo de ejecutar el benchmark con una configuración.

    Attributes:
        config: Configuración usada para este run.
        query_results: Resultado de cada query individual.
        aggregated: Métricas agregadas del run.
        errors: Lista de errores durante el run.
    """
    config: BenchmarkConfig
    query_results: list[QueryResult] = field(default_factory=list)
    aggregated: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    errors: list[str] = field(default_factory=list)
