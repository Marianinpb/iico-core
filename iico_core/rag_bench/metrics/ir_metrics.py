"""
iico_core/rag_bench/metrics/ir_metrics.py
==========================================
Métricas clásicas de Information Retrieval: Precision@K, Recall@K, MRR, nDCG.

Todas las funciones son puras (sin efectos laterales) y fácilmente testeables.

Referencia de fórmulas:
    - Precision@K = |retrieved ∩ relevant| / k
    - Recall@K    = |retrieved ∩ relevant| / |relevant|
    - MRR         = 1 / rank_of_first_relevant
    - nDCG        = DCG / IDCG  (Discounted Cumulative Gain normalizado)
"""

from __future__ import annotations

import math

from ..types import IRMetrics


def compute_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int | None = None,
) -> IRMetrics:
    """Calcula todas las métricas IR para una query.

    Args:
        retrieved_ids: IDs de chunks recuperados, **en orden de score** (mayor primero).
        relevant_ids: IDs de chunks relevantes (ground truth del dataset).
        k: Truncar la lista de recuperados a los primeros k. Si es None,
           usa ``len(retrieved_ids)`` (evalúa todos los recuperados).

    Returns:
        :class:`IRMetrics` con todas las métricas calculadas.
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids)
    relevant_retrieved = retrieved_set & relevant_set

    return IRMetrics(
        precision_at_k=_precision(retrieved_ids, relevant_set),
        recall_at_k=_recall(retrieved_ids, relevant_set, len(relevant_ids)),
        mrr=_mrr(retrieved_ids, relevant_set),
        ndcg=_ndcg(retrieved_ids, relevant_set),
        num_retrieved=len(retrieved_ids),
        num_relevant=len(relevant_ids),
        num_relevant_retrieved=len(relevant_retrieved),
    )


# ---------------------------------------------------------------------------
# Funciones individuales (expuestas para testing granular)
# ---------------------------------------------------------------------------

def _precision(retrieved: list[str], relevant: set[str]) -> float:
    """Precision@K: fracción de recuperados que son relevantes.

    ``P@K = |retrieved ∩ relevant| / K``
    """
    if not retrieved:
        return 0.0
    # Match si el relevant ID es idéntico o un substring del retrieved ID
    hits = sum(1 for r in retrieved if any(rel in r for rel in relevant))
    return hits / len(retrieved)


def _recall(retrieved: list[str], relevant: set[str], num_relevant: int) -> float:
    """Recall@K: fracción de relevantes que fueron recuperados.

    ``R@K = |retrieved ∩ relevant| / |relevant|``
    """
    if num_relevant == 0:
        return 0.0
    # Match si el relevant ID es idéntico o un substring del retrieved ID
    hits = sum(1 for r in retrieved if any(rel in r for rel in relevant))
    return hits / num_relevant


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: posición del primer chunk relevante.

    ``MRR = 1 / rank_of_first_relevant``
    Retorna 0 si ningún chunk recuperado es relevante.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if any(rel in chunk_id for rel in relevant):
            return 1.0 / rank
    return 0.0


def _dcg(retrieved: list[str], relevant: set[str]) -> float:
    """Discounted Cumulative Gain.

    ``DCG = Σ rel_i / log2(i + 1)``  donde ``rel_i = 1`` si el chunk en
    posición i es relevante, ``0`` si no.
    """
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if any(rel in chunk_id for rel in relevant):
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def _ideal_dcg(num_relevant: int, k: int) -> float:
    """IDCG: DCG del ranking ideal (todos los relevantes al principio)."""
    ideal_hits = min(num_relevant, k)
    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)
    return idcg


def _ndcg(retrieved: list[str], relevant: set[str]) -> float:
    """Normalized DCG.

    ``nDCG = DCG / IDCG``

    Retorna 0 si no hay elementos relevantes o si la lista está vacía.
    """
    if not retrieved or not relevant:
        return 0.0
    dcg = _dcg(retrieved, relevant)
    idcg = _ideal_dcg(len(relevant), len(retrieved))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
