"""
iico_core/rag_bench/metrics/performance.py
===========================================
Métricas de rendimiento: latencia, tokens de contexto, E_tok, y agregación.

Métrica central de la tesis — E_tok (Eficiencia de Tokens)::

    E_tok = total_context_tokens / (ragas_score + ε)

Interpretación:
    - Menor E_tok → más eficiente (menos tokens para igual o mejor calidad).
    - Permite comparar directamente dos chunkers sobre la misma query.
    - ε = 0.01 evita división por cero cuando ragas_score = 0.

Ejemplo de uso en la tesis::

    chunker A: 800 tokens, RAGAS=0.72 → E_tok = 800/0.73 = 1095
    chunker B: 400 tokens, RAGAS=0.70 → E_tok = 400/0.71 =  563

    chunker B es 48% más eficiente (casi misma calidad, mitad de tokens).
"""

from __future__ import annotations

import math

from ..types import AggregatedMetrics, PerformanceMetrics, QueryResult

# Épsilon para E_tok (evitar división por cero)
_E_TOK_EPS = 0.01


def compute_performance_metrics(
    retrieved_chunks: list[tuple[dict, float]],
    retrieval_latency_ms: float,
    ragas_score: float = 0.0,
    cache_hit: bool = False,
) -> PerformanceMetrics:
    """Calcula métricas de rendimiento para una query.

    Args:
        retrieved_chunks: Lista de ``(chunk_dict, score)`` recuperados.
        retrieval_latency_ms: Tiempo de recuperación en ms.
        ragas_score: Score RAGAS agregado (0.0 si no se calculó).
        cache_hit: True si el resultado vino del Splay Tree.

    Returns:
        :class:`PerformanceMetrics` con todas las métricas calculadas.
    """
    total_tokens = _count_context_tokens(retrieved_chunks)
    e_tok = _compute_e_tok(total_tokens, ragas_score)

    return PerformanceMetrics(
        retrieval_latency_ms=retrieval_latency_ms,
        total_context_tokens=total_tokens,
        ragas_score=ragas_score,
        e_tok=e_tok,
        cache_hit=cache_hit,
    )


def _count_context_tokens(chunks: list[tuple[dict, float]]) -> int:
    """Estima el número de tokens en todos los chunks recuperados.

    Estimación: 1 token ≈ 4 caracteres (aproximación estándar para LLMs).
    Consistente con la misma estimación usada en el chunker.

    Args:
        chunks: Lista de ``(chunk_dict, score)``.

    Returns:
        Total de tokens estimados.
    """
    total_chars = 0
    for chunk_dict, _ in chunks:
        content = chunk_dict.get("content", "")
        total_chars += len(content)
    return total_chars // 4


def _compute_e_tok(total_tokens: int, ragas_score: float) -> float:
    """Calcula la eficiencia de tokens.

    ``E_tok = total_context_tokens / (ragas_score + ε)``

    Retorna ``0.0`` si no hay tokens (contexto vacío).

    Args:
        total_tokens: Tokens totales en el contexto.
        ragas_score: Score RAGAS agregado.

    Returns:
        E_tok. Menor = más eficiente.
    """
    if total_tokens == 0:
        return 0.0
    return total_tokens / (ragas_score + _E_TOK_EPS)


def aggregate_metrics(query_results: list[QueryResult]) -> AggregatedMetrics:
    """Agrega métricas de todas las queries de un run.

    Calcula promedios de IR, performance y RAGAS sobre todos los
    ``QueryResult`` del run.

    Args:
        query_results: Lista de resultados individuales del run.

    Returns:
        :class:`AggregatedMetrics` con los promedios calculados.
    """
    if not query_results:
        return AggregatedMetrics()

    n = len(query_results)

    # Acumuladores performance
    sum_latency = sum(qr.perf.retrieval_latency_ms for qr in query_results)
    sum_tokens = sum(qr.perf.total_context_tokens for qr in query_results)
    sum_ragas = sum(qr.perf.ragas_score for qr in query_results)
    sum_e_tok = sum(qr.perf.e_tok for qr in query_results)

    sum_ragas_p = sum(qr.ragas.context_precision for qr in query_results if qr.ragas)
    sum_ragas_r = sum(qr.ragas.context_recall for qr in query_results if qr.ragas)

    # RAGAS caché hit rate
    ragas_results = [qr for qr in query_results if qr.ragas is not None]
    ragas_cache_hits = sum(1 for qr in ragas_results if qr.ragas and qr.ragas.from_cache)
    ragas_hit_rate = ragas_cache_hits / len(ragas_results) if ragas_results else 0.0

    # Splay hit rate
    splay_hits = sum(1 for qr in query_results if qr.perf.cache_hit)
    splay_hit_rate = splay_hits / n

    return AggregatedMetrics(
        num_queries=n,
        avg_latency_ms=sum_latency / n,
        avg_context_tokens=sum_tokens / n,
        avg_ragas_score=sum_ragas / n,
        avg_ragas_context_precision=sum_ragas_p / n,
        avg_ragas_context_recall=sum_ragas_r / n,
        avg_e_tok=sum_e_tok / n,
        ragas_cache_hit_rate=ragas_hit_rate,
        splay_hit_rate=splay_hit_rate,
    )
