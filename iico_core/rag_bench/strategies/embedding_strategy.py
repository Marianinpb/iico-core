"""
iico_core/rag_bench/strategies/embedding_strategy.py
======================================================
Estrategia de retrieval semántica basada en embeddings ONNX.

Flujo::

    setup()   → carga embeddings desde NoteDB (BLOB → np.ndarray)
              → construye matriz (n_chunks, 384) en RAM
              → tokeniza el índice una sola vez

    retrieve() → vectoriza el query con all-MiniLM-L6-v2
              → calcula dot product contra todos los embeddings (vectorizado)
              → filtra por threshold y retorna top-k

Es la estrategia baseline semántica del benchmark.
El :class:`SplayStrategy` usa esta misma clase internamente como fallback.

Config keys (en ``setup``):
    threshold (float): Umbral mínimo de cosine similarity. Default: ``0.0``
                       (sin filtro, retornar siempre top-k).
    model_name (str): Nombre del modelo registrado en NoteDB. Default: ``"all-MiniLM-L6-v2"``.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import RetrievalStrategy, register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class EmbeddingStrategy(RetrievalStrategy):
    """Búsqueda semántica vectorial sobre embeddings ONNX.

    Requiere que los chunks ya tengan embeddings en :class:`NoteDB`.
    Si la BD no tiene embeddings, llama al :class:`EmbeddingIndex` para
    generarlos on-the-fly (lento) o los omite.
    """

    name = "embeddings"
    description = "Búsqueda semántica con all-MiniLM-L6-v2 ONNX (cosine similarity)"

    def __init__(self) -> None:
        self._db: Any = None
        self._embedding_index: Any = None
        self._threshold: float = 0.0
        self._chunk_ids: list[str] = []
        self._chunk_map: dict[str, dict] = {}   # chunk_id → chunk_dict
        self._matrix: Any = None                # np.ndarray (n, 384) o None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        db: Any,
        embedding_index: Any,
        config: dict,
    ) -> None:
        """Carga embeddings desde NoteDB y construye la matriz en RAM.

        Args:
            db: instancia de :class:`NoteDB`.
            embedding_index: instancia de :class:`EmbeddingIndex`.
                             Se usa para generar embeddings de chunks que
                             aún no los tengan en la BD, y para vectorizar
                             el query en tiempo de retrieve.
            config:
                threshold (float): Umbral cosine. Default 0.0.
        """
        try:
            import numpy as np
            self._np = np
        except ImportError:
            raise ImportError(
                "numpy es requerido por EmbeddingStrategy. "
                "Instala: pip install iico-core[embeddings]"
            ) from None

        self._db = db
        self._embedding_index = embedding_index
        self._threshold = float(config.get("threshold", 0.0))
        self._setup_config = config

        self._load_index()

    def _load_index(self) -> None:
        """Carga embeddings de la BD y construye la matriz en RAM."""
        np = self._np
        strategy_name = self._setup_config.get("chunking_strategy_name")

        all_chunks = self._db.get_all_chunks(strategy_name)
        if not all_chunks:
            logger.warning(f"[EmbeddingStrategy] BD vacía para estrategia '{strategy_name}'.")
            self._matrix = None
            return

        # Cargar embeddings pre-computados desde NoteDB
        db_embeddings = self._db.load_all_embeddings(strategy_name)

        chunk_ids: list[str] = []
        embeddings: list[Any] = []
        missing: list[dict] = []

        for chunk in all_chunks:
            cid = chunk["id"]
            self._chunk_map[cid] = chunk
            if cid in db_embeddings:
                chunk_ids.append(cid)
                embeddings.append(db_embeddings[cid])
            else:
                missing.append(chunk)

        # Para chunks sin embedding: intentar generar on-the-fly
        if missing and self._embedding_index is not None:
            logger.info(
                "[EmbeddingStrategy] Generando embeddings para %d chunks sin vectorizar...",
                len(missing),
            )
            for chunk in missing:
                try:
                    text = self._build_chunk_text(chunk)
                    vec = self._embedding_index.vectorize(text)
                    # Guardar en BD para el próximo run
                    self._db.save_embedding(chunk["id"], vec)
                    chunk_ids.append(chunk["id"])
                    embeddings.append(vec)
                except Exception as e:
                    logger.warning(
                        "[EmbeddingStrategy] No se pudo vectorizar chunk '%s': %s",
                        chunk["id"], e,
                    )

        if not embeddings:
            logger.warning("[EmbeddingStrategy] Sin embeddings disponibles.")
            self._matrix = None
            return

        self._chunk_ids = chunk_ids
        self._matrix = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])

        logger.debug(
            "[EmbeddingStrategy] Índice construido: %d chunks, dim=%d",
            len(chunk_ids), self._matrix.shape[1],
        )

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Calcula cosine similarity query vs todos los chunks → top-k.

        Args:
            query: texto de la consulta.
            top_k: máximo de resultados.

        Returns:
            ``[(chunk_dict, score), ...]`` ordenado por score descendente.
            Si no hay embeddings o EmbeddingIndex, retorna lista vacía.
        """
        if self._matrix is None or self._embedding_index is None:
            logger.warning("[EmbeddingStrategy] Índice no disponible, retornando vacío.")
            return []

        np = self._np

        # Vectorizar el query
        try:
            query_vec = self._embedding_index.vectorize(query)
            query_vec = np.asarray(query_vec, dtype=np.float32)
        except Exception as e:
            logger.error("[EmbeddingStrategy] Error vectorizando query: %s", e)
            return []

        # Dot product vectorizado: (n, 384) @ (384,) → (n,)
        # Embeddings y query normalizados L2 → cosine similarity = dot product
        scores = self._matrix @ query_vec  # shape: (n,)

        # Filtrar por threshold y tomar top-k
        results: list[tuple[dict, float]] = []
        for i, score in enumerate(scores):
            score_f = float(score)
            if score_f >= self._threshold:
                cid = self._chunk_ids[i]
                chunk_dict = self._chunk_map.get(cid)
                if chunk_dict is not None:
                    results.append((chunk_dict, score_f))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chunk_text(chunk: dict) -> str:
        """Construye el texto a vectorizar para un chunk (mismo formato que EmbeddingIndex)."""
        tags = chunk.get("tags", [])
        if isinstance(tags, list):
            tags_str = " ".join(tags)
        else:
            tags_str = str(tags)
        content = chunk.get("content", "")[:512]
        return f"{chunk['id']} {tags_str} {content}"

    @property
    def index_size(self) -> int:
        """Número de chunks en el índice."""
        return len(self._chunk_ids)

    def rebuild(self) -> None:
        """Re-construye el índice desde la BD (útil tras re-chunking)."""
        self._chunk_ids = []
        self._chunk_map = {}
        self._matrix = None
        self._load_index()
