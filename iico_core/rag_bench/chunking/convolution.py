"""
iico_core/rag_bench/chunking/convolution.py
============================================
Chunker experimental: Convolución 1D sobre embeddings de oraciones/párrafos.

Algoritmo
---------
1. **Segmentar** el texto en unidades (oraciones o párrafos).
2. **Vectorizar** cada unidad con el EmbeddingIndex (ONNX).
3. **Señal de similaridad**: ``sim[i] = cosine(e[i], e[i+1])`` → señal 1D.
4. **Convolución 1D**: ``smoothed = conv1d(sim, kernel)`` con kernel configurable.
5. **Detección de valles**: mínimos locales bajo el umbral → puntos de corte.
6. **Exportar señales DSP** en ``ChunkingResult.metadata`` para correlogramas.

Kernels disponibles
-------------------
- ``gaussian``: Suavizado gradual. Detecta cambios de tema suaves.
- ``edge_detect``: Derivada discreta. Detecta cortes abruptos.
- ``box``: Promedio uniforme. Baseline simple.
- ``custom``: Pesos arbitrarios provistos por el usuario.

Señales exportadas (para gráficas de tesis)
-------------------------------------------
El campo ``metadata`` del ``ChunkingResult`` incluye::

    {
        "chunker": "convolution",
        "kernel_type": "gaussian",
        "kernel_size": 5,
        "kernel_weights": [...],
        "unit": "sentence",
        "valley_threshold": 0.5,
        "units": ["oración 1", ...],
        "num_units": 8,
        "similarity_signal": [0.82, 0.61, 0.29, 0.88, ...],  # raw
        "smoothed_signal": [0.78, 0.55, 0.38, 0.82, ...],    # después de conv
        "valley_positions": [2, 5],
        "valley_depths": [0.38, 0.31],                        # smoothed[valley]
        "boundary_confidences": [0.62, 0.69],
    }

Ejemplo de uso
--------------
::

    chunker = ConvolutionChunker()
    chunker.setup(embedding_index=my_index)
    result = chunker.chunk(text, config={
        "kernel_type": "gaussian",
        "kernel_size": 5,
        "valley_threshold": 0.5,
        "unit": "sentence",
    })
    # result.metadata["similarity_signal"] → graficar en tesis
    # result.fragments → chunks resultantes
"""

from __future__ import annotations

import re
from typing import Any

from .base import ChunkBoundary, ChunkingResult, ChunkingStrategy, register_chunker

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# ConvolutionChunker
# ---------------------------------------------------------------------------

@register_chunker
class ConvolutionChunker(ChunkingStrategy):
    """Chunker semántico por convolución 1D sobre embeddings.

    Requiere ``setup(embedding_index=...)`` antes de llamar a ``chunk()``.
    Si el EmbeddingIndex no está disponible (test unitario sin ONNX),
    degrada graciosamente a un chunker por párrafos.

    Config keys:
        kernel_type (str): ``"gaussian"`` | ``"edge_detect"`` | ``"box"`` | ``"custom"``.
                           Default: ``"gaussian"``.
        kernel_size (int): Tamaño del kernel (impar recomendado). Default: ``5``.
        kernel_weights (list[float]): Pesos del kernel custom. Solo con ``kernel_type="custom"``.
        valley_threshold (float): Umbral [0,1]. Posiciones donde ``smoothed < threshold``
                                  se consideran puntos de corte. Default: ``0.5``.
        unit (str): Granularidad de embedding: ``"sentence"`` | ``"paragraph"``.
                    Default: ``"sentence"``.
        min_chunk_chars (int): Fragmentos más pequeños se fusionan. Default: ``30``.
    """

    name = "convolution"
    description = "Convolución 1D sobre embeddings para detectar fronteras semánticas"

    def __init__(self) -> None:
        self._embedding_index: Any = None

    def setup(self, embedding_index: Any = None, **kwargs: Any) -> None:
        self._embedding_index = embedding_index

    def chunk(self, text: str, config: dict | None = None) -> ChunkingResult:
        cfg = config or {}
        kernel_type: str = cfg.get("kernel_type", "gaussian")
        kernel_size: int = cfg.get("kernel_size", 5)
        kernel_weights: list | None = cfg.get("kernel_weights", None)
        threshold: float = cfg.get("valley_threshold", 0.5)
        unit: str = cfg.get("unit", "sentence")
        min_chars: int = cfg.get("min_chunk_chars", 30)

        # ── Segmentar ──────────────────────────────────────────────────
        units = self._segment(text, unit)

        if not units:
            return ChunkingResult(
                fragments=[text.strip()] if text.strip() else [],
                boundaries=[],
                metadata={"chunker": "convolution", "error": "empty_text"},
            )

        # ── Vectorizar (o degradar) ────────────────────────────────────
        embeddings = self._vectorize_units(units)

        # Si no pudimos vectorizar (sin EmbeddingIndex o numpy), degradar
        if embeddings is None:
            return self._fallback_paragraph_split(text, cfg)

        # ── Señal de similaridad ──────────────────────────────────────
        sim_signal = self._similarity_signal(embeddings)

        if len(sim_signal) == 0:
            # Solo 1 unidad: no hay cortes posibles
            return ChunkingResult(
                fragments=[text.strip()],
                boundaries=[],
                metadata={
                    "chunker": "convolution",
                    "kernel_type": kernel_type,
                    "kernel_size": kernel_size,
                    "unit": unit,
                    "num_units": len(units),
                    "similarity_signal": [],
                    "smoothed_signal": [],
                    "valley_positions": [],
                    "valley_depths": [],
                    "boundary_confidences": [],
                },
            )

        # ── Convolución 1D ────────────────────────────────────────────
        kernel = self._build_kernel(kernel_type, kernel_size, kernel_weights)
        smoothed = np.convolve(sim_signal, kernel, mode="same")

        # Clipear a [0, 1] (el suavizado puede introducir valores fuera)
        smoothed = np.clip(smoothed, 0.0, 1.0)

        # ── Detección de valles ───────────────────────────────────────
        valley_positions = self._detect_valleys(smoothed, threshold)

        # ── Construir fragmentos ──────────────────────────────────────
        fragments, boundaries = self._build_fragments(
            units, valley_positions, smoothed, min_chars
        )

        # ── Empaquetar metadata DSP (para correlogramas de tesis) ─────
        metadata = self._build_dsp_metadata(
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_weights=kernel.tolist(),
            unit=unit,
            threshold=threshold,
            units=units,
            sim_signal=sim_signal,
            smoothed=smoothed,
            valley_positions=valley_positions,
            boundaries=boundaries,
        )

        return ChunkingResult(
            fragments=fragments,
            boundaries=boundaries,
            metadata=metadata,
        )

    # ==================================================================
    # Kernels
    # ==================================================================

    def _build_kernel(
        self,
        kernel_type: str,
        kernel_size: int,
        custom_weights: list | None,
    ) -> "np.ndarray":
        """Construye el kernel de convolución según el tipo."""
        if kernel_type == "gaussian":
            return self._gaussian_kernel(kernel_size)
        elif kernel_type == "edge_detect":
            return self._edge_kernel(kernel_size)
        elif kernel_type == "box":
            return np.ones(kernel_size, dtype=np.float64) / kernel_size
        elif kernel_type == "custom":
            if custom_weights is None:
                raise ValueError("kernel_type='custom' requiere kernel_weights.")
            k = np.array(custom_weights, dtype=np.float64)
            total = np.sum(np.abs(k))
            return k / total if total > 0 else k
        else:
            raise ValueError(
                f"kernel_type='{kernel_type}' desconocido. "
                "Opciones: gaussian, edge_detect, box, custom."
            )

    @staticmethod
    def _gaussian_kernel(size: int) -> "np.ndarray":
        """Kernel gaussiano normalizado.

        σ = size/4. Suaviza gradualmente; detecta cambios de tema lentos.
        Ideal para textos con transiciones suaves entre párrafos.
        """
        sigma = max(size / 4.0, 0.5)
        x = np.arange(size, dtype=np.float64) - size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / kernel.sum()

    @staticmethod
    def _edge_kernel(size: int) -> "np.ndarray":
        """Kernel de detección de bordes (primera derivada discreta).

        Derivada simétrica: parte negativa → positiva.
        Detecta cambios abruptos. Para textos con secciones claramente delimitadas.
        """
        kernel = np.zeros(size, dtype=np.float64)
        half = size // 2
        kernel[:half] = -1.0
        kernel[half + 1:] = 1.0
        total = np.sum(np.abs(kernel))
        return kernel / total if total > 0 else kernel

    # ==================================================================
    # Segmentación
    # ==================================================================

    @staticmethod
    def _segment(text: str, unit: str) -> list[str]:
        """Segmenta el texto en oraciones o párrafos."""
        if unit == "paragraph":
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        else:  # "sentence"
            # Divide en oraciones por `.`, `!`, `?` seguidos de espacio/newline
            raw = re.split(r"(?<=[.!?])\s+", text)
            parts = [s.strip() for s in raw if s.strip()]

        return parts

    # ==================================================================
    # Vectorización
    # ==================================================================

    def _vectorize_units(self, units: list[str]) -> "np.ndarray | None":
        """Vectoriza las unidades con el EmbeddingIndex.

        Returns:
            Array de shape ``(n_units, dim)`` con embeddings L2-normalizados,
            o ``None`` si no hay EmbeddingIndex o numpy disponibles.
        """
        if not _HAS_NUMPY or self._embedding_index is None:
            return None

        embeddings: list[np.ndarray] = []
        for unit in units:
            try:
                vec = self._embedding_index.vectorize(unit)
                vec = np.asarray(vec, dtype=np.float64)
                # Normalizar a L2 para que el dot product = cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)
            except Exception:
                return None  # Un fallo → degradar

        if not embeddings:
            return None

        return np.stack(embeddings)

    # ==================================================================
    # Señal de similaridad
    # ==================================================================

    @staticmethod
    def _similarity_signal(embeddings: "np.ndarray") -> "np.ndarray":
        """Cosine similarity entre embeddings consecutivos → señal 1D.

        ``sim[i] = dot(e[i], e[i+1])``  (e ya normalizado → cosine)

        Returns:
            Array de shape ``(n-1,)`` con valores en ``[-1, 1]``.
        """
        n = len(embeddings)
        if n < 2:
            return np.array([], dtype=np.float64)
        sims = []
        for i in range(n - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            sims.append(sim)
        return np.array(sims, dtype=np.float64)

    # ==================================================================
    # Detección de valles
    # ==================================================================

    @staticmethod
    def _detect_valleys(signal: "np.ndarray", threshold: float) -> list[int]:
        """Detecta mínimos locales que están bajo el umbral.

        Un valle en posición ``i`` significa que ``signal[i]`` es menor
        que sus vecinos Y menor que el umbral. El corte ocurre DESPUÉS
        de la unidad ``i``.

        Returns:
            Lista de posiciones (índices) de valles detectados.
        """
        n = len(signal)
        valleys = []

        for i in range(n):
            if signal[i] >= threshold:
                continue

            # Verificar que es mínimo local
            left_ok = (i == 0) or (signal[i] <= signal[i - 1])
            right_ok = (i == n - 1) or (signal[i] <= signal[i + 1])

            if left_ok and right_ok:
                valleys.append(i)

        return valleys

    # ==================================================================
    # Construcción de fragmentos
    # ==================================================================

    @staticmethod
    def _build_fragments(
        units: list[str],
        valley_positions: list[int],
        smoothed: "np.ndarray",
        min_chars: int,
    ) -> tuple[list[str], list[ChunkBoundary]]:
        """Construye fragmentos a partir de valles detectados.

        Cada valle en posición ``p`` = corte después de la unidad ``p``.
        Las unidades 0..p van al fragmento anterior; p+1..next_valley al siguiente.

        Fragmentos más pequeños que ``min_chars`` se fusionan con el anterior.
        """
        if not valley_positions:
            return [" ".join(units).strip()], []

        # Puntos de corte: slice de unidades
        cut_points = sorted(valley_positions)
        fragments_raw: list[list[str]] = []
        start = 0

        for cut in cut_points:
            fragment_units = units[start: cut + 1]
            if fragment_units:
                fragments_raw.append(fragment_units)
            start = cut + 1

        # Último fragmento
        if start < len(units):
            fragments_raw.append(units[start:])

        # Unir unidades en texto y filtrar por min_chars
        boundaries: list[ChunkBoundary] = []
        fragments: list[str] = []
        char_cursor = 0

        for i, unit_group in enumerate(fragments_raw):
            text = " ".join(unit_group).strip()
            if not text:
                continue

            if fragments and len(text) < min_chars:
                # Fusionar con el fragmento anterior
                fragments[-1] = fragments[-1] + " " + text
                char_cursor += len(text)
                continue

            if fragments:
                # El corte corresponde al valley en cut_points[i-1]
                valley_idx = cut_points[i - 1] if i - 1 < len(cut_points) else cut_points[-1]
                depth = float(smoothed[valley_idx]) if valley_idx < len(smoothed) else 0.5
                boundaries.append(ChunkBoundary(
                    position=char_cursor,
                    confidence=max(0.0, min(1.0, 1.0 - depth)),
                    reason=f"kernel_valley(depth={depth:.3f})",
                ))

            fragments.append(text)
            char_cursor += len(text)

        if not fragments:
            fragments = [" ".join(units).strip()]

        return fragments, boundaries

    # ==================================================================
    # Metadata DSP
    # ==================================================================

    @staticmethod
    def _build_dsp_metadata(
        kernel_type: str,
        kernel_size: int,
        kernel_weights: list[float],
        unit: str,
        threshold: float,
        units: list[str],
        sim_signal: "np.ndarray",
        smoothed: "np.ndarray",
        valley_positions: list[int],
        boundaries: list[ChunkBoundary],
    ) -> dict[str, Any]:
        """Construye el dict de metadata DSP para el ReportGenerator.

        Todas las señales se exportan como listas de Python (JSON-serializable).
        """
        return {
            # Identificación
            "chunker": "convolution",
            "kernel_type": kernel_type,
            "kernel_size": kernel_size,
            "kernel_weights": kernel_weights,
            "unit": unit,
            "valley_threshold": threshold,
            # Conteos
            "num_units": len(units),
            "num_valleys": len(valley_positions),
            # ── Señales DSP (para correlogramas de tesis) ──────────────
            "similarity_signal": sim_signal.tolist(),          # señal cruda
            "smoothed_signal": smoothed.tolist(),              # después de conv
            "valley_positions": valley_positions,              # índices de valles
            "valley_depths": [
                float(smoothed[p]) for p in valley_positions
                if p < len(smoothed)
            ],
            "boundary_confidences": [b.confidence for b in boundaries],
            # Texto de las unidades (para debug, omitir en reportes masivos)
            "units_preview": [u[:60] + "..." if len(u) > 60 else u for u in units],
        }

    # ==================================================================
    # Fallback sin EmbeddingIndex
    # ==================================================================

    def _fallback_paragraph_split(
        self, text: str, cfg: dict
    ) -> ChunkingResult:
        """Fallback sin EmbeddingIndex: divide por párrafos dobles."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        boundaries = [
            ChunkBoundary(
                position=sum(len(p) for p in paragraphs[:i]),
                confidence=0.5,
                reason="paragraph_break_fallback",
            )
            for i in range(1, len(paragraphs))
        ]

        return ChunkingResult(
            fragments=paragraphs,
            boundaries=boundaries,
            metadata={
                "chunker": "convolution",
                "fallback": True,
                "reason": "no_embedding_index",
                "similarity_signal": [],
                "smoothed_signal": [],
                "valley_positions": [],
                "valley_depths": [],
                "boundary_confidences": [],
            },
        )
