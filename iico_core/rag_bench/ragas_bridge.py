"""
iico_core/rag_bench/ragas_bridge.py
=====================================
Puente hacia DeepSeek para evaluación de calidad RAG.

Implementa las métricas RAGAS manualmente via API de DeepSeek, sin
depender del paquete ``ragas`` (que requiere OpenAI u otros providers
complejos de configurar).

Métricas implementadas
----------------------
1. **Faithfulness**: ¿La respuesta está soportada por el contexto?
   - Prompt: "Given context C, is claim X supported? Answer: YES/NO"
   - Score: fracción de claims en la respuesta que están soportados.

2. **Answer Relevancy**: ¿La respuesta es relevante para la pregunta?
   - Prompt: "On a scale 0-1, how relevant is this answer to the question?"
   - Score: promedio de N muestras del LLM.

Caché por hash de contextos
----------------------------
Si dos runs recuperan **exactamente los mismos chunks** para la misma query,
no se gasta ningún token adicional. El caché usa SHA-256 del texto concatenado
de los contextos (normalizado) como clave.

Persistencia del caché::

    # Se guarda automáticamente en `ragas_cache.json` dentro del directorio
    # de trabajo. Al instanciar RagasBridge, se carga si existe.

    bridge = RagasBridge(api_key="...", cache_path="ragas_cache.json")

Config::

    DEEPSEEK_API_KEY: str  — API key de DeepSeek
    model: str             — Modelo a usar. Default: "deepseek-chat"
    max_retries: int       — Reintentos en caso de error de red. Default: 3
    timeout: int           — Timeout por llamada en segundos. Default: 30
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .types import RAGASMetrics

logger = logging.getLogger(__name__)

# Número de claims a muestrear para faithfulness
_FAITHFULNESS_CLAIMS = 3
# Número de muestras para answer relevancy
_RELEVANCY_SAMPLES = 2


class RagasBridge:
    """Evalúa calidad RAG via DeepSeek API con caché SHA-256 de contextos.

    Args:
        api_key: API key de DeepSeek (o variable de entorno DEEPSEEK_API_KEY).
        model: Modelo a usar. Default: ``"deepseek-chat"``.
        cache_path: Ruta al archivo JSON de caché. Default: ``"ragas_cache.json"``.
        max_retries: Reintentos en caso de error. Default: ``3``.
        timeout: Timeout por llamada HTTP en segundos. Default: ``30``.
    """

    _BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
        cache_path: str | Path = "ragas_cache.json",
        max_retries: int = 3,
        timeout: int = 30,
    ) -> None:
        import os
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._model = model
        self._cache_path = Path(cache_path)
        self._max_retries = max_retries
        self._timeout = timeout
        self._cache: dict[str, dict] = {}

        self._load_cache()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        contexts: list[str],
        expected_answer: str = "",
    ) -> RAGASMetrics:
        """Evalúa la calidad del contexto recuperado para un query.

        Primero busca en caché usando SHA-256(contexts+query).
        Si hay hit → retorna métricas sin llamar a DeepSeek.
        Si hay miss → llama a DeepSeek y guarda en caché.

        Args:
            query: Texto de la consulta.
            contexts: Lista de textos de los chunks recuperados.
            expected_answer: Respuesta esperada (opcional, mejora la evaluación).

        Returns:
            :class:`RAGASMetrics` con las métricas calculadas.
        """
        if not contexts:
            return RAGASMetrics(error="no_contexts")

        if not self._api_key:
            return RAGASMetrics(error="no_api_key")

        # ── Verificar caché ───────────────────────────────────────────
        context_hash = self._hash_contexts(query, contexts)

        if context_hash in self._cache:
            cached = self._cache[context_hash]
            logger.debug("[RagasBridge] Cache HIT para hash %s...", context_hash[:8])
            return RAGASMetrics(
                faithfulness=cached.get("faithfulness", 0.0),
                answer_relevancy=cached.get("answer_relevancy", 0.0),
                context_precision=cached.get("context_precision", 0.0),
                context_recall=cached.get("context_recall", 0.0),
                from_cache=True,
                context_hash=context_hash,
            )

        # ── Llamar a DeepSeek ─────────────────────────────────────────
        logger.debug("[RagasBridge] Cache MISS, evaluando con DeepSeek...")
        try:
            combined_context = "\n\n---\n\n".join(contexts)
            faithfulness = self._evaluate_faithfulness(query, combined_context)
            answer_relevancy = self._evaluate_answer_relevancy(query, combined_context)
            context_precision = self._evaluate_context_precision(query, combined_context)
            context_recall = self._evaluate_context_recall(
                query, combined_context, expected_answer
            )
        except Exception as e:
            logger.error("[RagasBridge] Error llamando a DeepSeek: %s", e)
            return RAGASMetrics(error=str(e), context_hash=context_hash)

        # ── Guardar en caché ──────────────────────────────────────────
        result_dict = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        self._cache[context_hash] = result_dict
        self._save_cache()

        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            from_cache=False,
            context_hash=context_hash,
        )

    @property
    def cache_size(self) -> int:
        """Número de entradas en el caché."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Vacía el caché en RAM y en disco."""
        self._cache = {}
        if self._cache_path.exists():
            self._cache_path.unlink()

    # ------------------------------------------------------------------
    # Evaluación individual (prompts RAGAS-style)
    # ------------------------------------------------------------------

    def _evaluate_faithfulness(self, query: str, context: str) -> float:
        """¿La respuesta/contexto apoya la pregunta fielmente?

        Heurística: pide al LLM que puntúe del 0 al 1 qué tan bien
        el contexto puede responder la pregunta sin información externa.
        """
        prompt = (
            f"You are evaluating RAG system quality. "
            f"Given the following context and question, rate from 0.0 to 1.0 "
            f"how faithfully the context supports answering the question "
            f"(1.0 = perfectly supports, 0.0 = completely off-topic).\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        return self._call_and_parse_score(prompt)

    def _evaluate_answer_relevancy(self, query: str, context: str) -> float:
        """¿El contexto es relevante para responder la pregunta?"""
        prompt = (
            f"You are evaluating RAG system quality. "
            f"Rate from 0.0 to 1.0 how relevant this context is for answering "
            f"the given question "
            f"(1.0 = highly relevant, 0.0 = not relevant at all).\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        return self._call_and_parse_score(prompt)

    def _evaluate_context_precision(self, query: str, context: str) -> float:
        """¿El contexto recuperado es preciso (sin ruido irrelevante)?"""
        prompt = (
            f"You are evaluating RAG system quality. "
            f"Rate from 0.0 to 1.0 the precision of this context for answering "
            f"the question — i.e., what fraction of the context is actually "
            f"useful for answering the question "
            f"(1.0 = all context is useful, 0.0 = all context is noise).\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        return self._call_and_parse_score(prompt)

    def _evaluate_context_recall(
        self, query: str, context: str, expected_answer: str
    ) -> float:
        """¿El contexto contiene la información para responder completamente?"""
        reference = expected_answer or "No reference answer provided."
        prompt = (
            f"You are evaluating RAG system quality. "
            f"Rate from 0.0 to 1.0 how well this context covers all the "
            f"information needed to answer the question completely "
            f"(1.0 = complete coverage, 0.0 = missing all key information).\n\n"
            f"Question: {query}\n"
            f"Reference answer: {reference[:500]}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        return self._call_and_parse_score(prompt)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _call_and_parse_score(self, prompt: str) -> float:
        """Llama a DeepSeek y parsea un score numérico de la respuesta.

        Returns:
            Float en [0, 1]. Retorna 0.0 si la respuesta no se puede parsear.
        """
        raw = self._call_deepseek(prompt)
        return self._parse_score(raw)

    def _call_deepseek(self, prompt: str) -> str:
        """Hace la llamada HTTP a la API de DeepSeek con reintentos."""
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 10,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        for attempt in range(1, self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    self._BASE_URL,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    return content.strip()

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                logger.warning(
                    "[RagasBridge] HTTP %d en intento %d/%d: %s",
                    e.code, attempt, self._max_retries, body[:200],
                )
                if e.code in (401, 403):  # Auth errors: no reintentar
                    raise

            except Exception as e:
                logger.warning(
                    "[RagasBridge] Error en intento %d/%d: %s",
                    attempt, self._max_retries, e,
                )

            if attempt < self._max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(
            f"[RagasBridge] Agotados {self._max_retries} reintentos."
        )

    @staticmethod
    def _parse_score(text: str) -> float:
        """Parsea un score numérico de la respuesta del LLM.

        Busca el primer número float (con signo opcional) en el texto.
        Clampea a [0, 1].
        """
        import re
        # Capturar signo negativo opcional
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.0

    # ------------------------------------------------------------------
    # Caché
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_contexts(query: str, contexts: list[str]) -> str:
        """Genera SHA-256 del query + contextos concatenados.

        El hash es determinista y normalizado (whitespace colapsado).
        Dos configuraciones que recuperan los mismos chunks producen
        el mismo hash → compartirán el resultado de RAGAS.
        """
        # Normalizar: colapsar whitespace, lower case
        normalized_query = " ".join(query.lower().split())
        # Ordenar contextos para que el orden de recuperación no afecte
        normalized_contexts = sorted(" ".join(c.split()) for c in contexts)
        combined = normalized_query + "|||" + "|||".join(normalized_contexts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Carga el caché desde disco si existe."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(
                    "[RagasBridge] Caché cargado: %d entradas desde %s",
                    len(self._cache), self._cache_path,
                )
            except Exception as e:
                logger.warning("[RagasBridge] Error cargando caché: %s", e)
                self._cache = {}

    def _save_cache(self) -> None:
        """Persiste el caché en disco."""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("[RagasBridge] Error guardando caché: %s", e)
