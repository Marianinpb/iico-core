"""
iico_core/rag_bench/strategies/splay_strategy.py
=================================================
Estrategia de retrieval que simula la arquitectura dual del Harness:
Splay Tree (Nivel 2) → miss → EmbeddingIndex (Nivel 1).

Propósito en la tesis
----------------------
Medir el **beneficio real del caché Splay** comparado con búsqueda semántica
pura. En un benchmark con múltiples queries secuenciales sobre el mismo corpus,
el Splay debería mostrar:

- **Hit rate creciente**: con el tiempo más queries se resuelven desde caché.
- **Latencia decreciente**: hits del Splay < 1ms vs ~5-50ms de embeddings.
- **Convergencia del árbol**: avg_depth decrece con el número de accesos.

Flujo exacto (idéntico al Harness)::

    1. peek_top(n) → revisar los n nodos más cerca de la raíz (sin splayear)
       Si el query tiene tokens en las keys top → HIT rápido
    2. Miss → EmbeddingStrategy.retrieve()
    3. Insertar resultado en el Splay Tree para futuras queries
    4. SplayCacheMetrics registra hit/miss y profundidad

Config keys (en ``setup``):
    splay_cache_size (int): Capacidad máxima del caché. Default: ``50``.
    peek_n (int): Cuántos nodos revisar en peek_top(). Default: ``5``.
    embedding_threshold (float): Umbral para el fallback de embeddings. Default: ``0.0``.
    hit_strategy (str): ``"token_overlap"`` (default) | ``"exact_match"``.
                        Cómo decidir si hay hit en peek_top.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import RetrievalStrategy, register_strategy
from .embedding_strategy import EmbeddingStrategy

logger = logging.getLogger(__name__)


@register_strategy
class SplayStrategy(RetrievalStrategy):
    """Splay Tree cache + EmbeddingStrategy fallback.

    Simula la arquitectura dual de memoria del Harness para benchmarking.
    Permite medir el impacto del Splay en términos de hit rate y latencia.
    """

    name = "splay"
    description = "Splay Tree caché (Nivel 2) + Embedding fallback (Nivel 1)"

    def __init__(self) -> None:
        self._embedding_strategy = EmbeddingStrategy()
        self._splay: Any = None                      # SplayTree
        self._metrics: Any = None                    # SplayCacheMetrics
        self._peek_n: int = 5
        self._hit_strategy: str = "token_overlap"
        self._db: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        db: Any,
        embedding_index: Any,
        config: dict,
    ) -> None:
        """Inicializa el Splay Tree y el EmbeddingStrategy subyacente.

        Args:
            db: instancia de :class:`NoteDB`.
            embedding_index: instancia de :class:`EmbeddingIndex`.
            config:
                splay_cache_size (int): capacidad máxima del caché. Default 50.
                peek_n (int): nodos a revisar en peek_top. Default 5.
                embedding_threshold (float): umbral del fallback. Default 0.0.
                hit_strategy (str): "token_overlap" | "exact_match". Default "token_overlap".
        """
        self._db = db
        self._peek_n = int(config.get("peek_n", 5))
        self._hit_strategy = config.get("hit_strategy", "token_overlap")

        # Inicializar el Splay Tree
        try:
            from ...index.splay_tree import SplayTree, SplayCacheMetrics
            cache_size = int(config.get("splay_cache_size", 50))
            self._metrics = SplayCacheMetrics()
            self._splay = SplayTree(max_nodes=cache_size, metrics=self._metrics)
            logger.debug("[SplayStrategy] SplayTree inicializado, max_nodes=%d", cache_size)
        except ImportError:
            logger.error("[SplayStrategy] No se pudo importar SplayTree. Usando solo embeddings.")
            self._splay = None

        # Inicializar el EmbeddingStrategy (fallback)
        emb_config = {
            "threshold": config.get("embedding_threshold", 0.0),
            "chunking_strategy_name": config.get("chunking_strategy_name"),
        }
        self._embedding_strategy.setup(db, embedding_index, emb_config)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Busca en el Splay Tree; si falla, delega a EmbeddingStrategy.

        Args:
            query: texto de la consulta.
            top_k: máximo de resultados.

        Returns:
            ``[(chunk_dict, score), ...]`` ordenado por score descendente.
        """
        # Si no hay Splay disponible, usar solo embeddings
        if self._splay is None:
            return self._embedding_strategy.retrieve(query, top_k)

        query_tokens = self._tokenize(query)

        # ── Paso 1: peek_top sin splayear (O(peek_n)) ─────────────────
        top_nodes = self._splay.peek_top(self._peek_n)
        hit_chunks = self._check_hit(query_tokens, top_nodes)

        if hit_chunks:
            # HIT: retornar desde caché (ya están cerca de la raíz)
            logger.debug("[SplayStrategy] HIT en caché para query: %.50s", query)
            return hit_chunks[:top_k]

        # ── Paso 2: MISS → delegar a EmbeddingStrategy ────────────────
        logger.debug("[SplayStrategy] MISS, delegando a EmbeddingStrategy...")
        results = self._embedding_strategy.retrieve(query, top_k)

        # ── Paso 3: Insertar resultados en el Splay ────────────────────
        if results and self._splay is not None:
            # Clave del caché: tokens del query (normalizado)
            cache_key = self._build_cache_key(query_tokens)
            self._splay.insert(cache_key, results)
            logger.debug("[SplayStrategy] Insertado en caché: key='%s'", cache_key)

        return results

    # ------------------------------------------------------------------
    # Lógica de hit check
    # ------------------------------------------------------------------

    def _check_hit(
        self,
        query_tokens: set[str],
        top_nodes: list,
    ) -> list[tuple[dict, float]]:
        """Verifica si algún nodo del peek_top es relevante para el query.

        Estrategias:
        - ``token_overlap``: el nodo tiene ≥ 1 token en común con el query.
        - ``exact_match``: la key del nodo es exactamente el query normalizado.

        Returns:
            Lista de ``(chunk_dict, score)`` del primer nodo con hit, o ``[]``.
        """
        if not top_nodes or not query_tokens:
            return []

        for node in top_nodes:
            if self._hit_strategy == "exact_match":
                cache_key = self._build_cache_key(query_tokens)
                if node.key == cache_key:
                    return list(node.value)
            else:  # token_overlap
                node_tokens = set(node.key.split())
                if query_tokens & node_tokens:  # Intersección no vacía
                    return list(node.value)

        return []

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokeniza el texto en palabras ≥ 2 caracteres, normalizadas."""
        words = re.findall(r"\b\w{2,}\b", text.lower())
        return set(words)

    @staticmethod
    def _build_cache_key(tokens: set[str]) -> str:
        """Construye una key de caché estable a partir de tokens."""
        return " ".join(sorted(tokens))

    # ------------------------------------------------------------------
    # Métricas para reportes
    # ------------------------------------------------------------------

    @property
    def cache_metrics(self) -> dict:
        """Métricas del Splay Tree para el ReportGenerator.

        Returns::

            {
                "hits": int,
                "misses": int,
                "total_accesses": int,
                "hit_rate": float,        # 0.0 - 1.0
                "avg_depth": float,       # profundidad promedio de acceso
                "depth_history": [        # para graficar convergencia
                    (n_acceso, avg_depth), ...
                ],
            }
        """
        if self._metrics is None:
            return {
                "hits": 0, "misses": 0, "total_accesses": 0,
                "hit_rate": 0.0, "avg_depth": 0.0, "depth_history": [],
            }
        summary = self._metrics.summary()
        summary["depth_history"] = self._metrics.depth_history
        return summary

    def reset_cache(self) -> None:
        """Vacía el caché Splay y resetea métricas.

        Útil entre runs del benchmark para aislar el efecto del caché.
        """
        if self._splay is not None:
            try:
                from ...index.splay_tree import SplayTree, SplayCacheMetrics
                cache_size = self._splay.max_nodes
                self._metrics = SplayCacheMetrics()
                self._splay = SplayTree(max_nodes=cache_size, metrics=self._metrics)
            except Exception:
                pass

    def teardown(self) -> None:
        self.reset_cache()
        self._embedding_strategy.teardown()
