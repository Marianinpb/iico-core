"""
iico_core/rag_bench/chunking/pipeline.py
==========================================
Composición de múltiples :class:`ChunkingStrategy` en secuencia.

Cada etapa recibe los fragmentos de la anterior y los refina.
Las señales DSP de todas las etapas se consolidan en el resultado final.

Ejemplo::

    pipeline = ChunkingPipeline([
        ("structural", {}),
        ("convolution", {"kernel_type": "gaussian", "kernel_size": 5}),
    ])
    pipeline.setup(embedding_index=my_index)
    result = pipeline.run(text)
    # result.metadata["stages"]["convolution"]["similarity_signal"] → graficar
"""

from __future__ import annotations

from typing import Any

from .base import ChunkBoundary, ChunkingResult, get_chunker, ChunkingStrategy


class ChunkingPipeline:
    """Encadena múltiples :class:`ChunkingStrategy` en secuencia.

    **Primera etapa**: recibe el texto completo y produce fragmentos iniciales.

    **Etapas siguientes**: cada fragmento pasa por la siguiente estrategia
    individualmente. Los sub-fragmentos se concatenan en el resultado final.

    Las señales DSP de cada etapa se guardan en ``result.metadata["stages"]``
    para que el :class:`ReportGenerator` pueda exportarlas por separado.

    Args:
        stages: Lista de tuplas ``(nombre_chunker, config_dict)``.
                El nombre debe estar registrado en el registry.

    Example::

        pipeline = ChunkingPipeline([
            ("structural", {"max_chunk_tokens": 512}),
            ("convolution", {
                "kernel_type": "gaussian",
                "kernel_size": 5,
                "valley_threshold": 0.45,
            }),
        ])
    """

    def __init__(self, stages: list[tuple[str, dict]]) -> None:
        if not stages:
            raise ValueError("ChunkingPipeline requiere al menos 1 etapa.")

        self._stage_configs: list[tuple[str, dict]] = stages
        self._chunkers: list[tuple[ChunkingStrategy, dict]] = []

        for name, config in stages:
            cls = get_chunker(name)
            self._chunkers.append((cls(), config))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, embedding_index: Any = None, **kwargs: Any) -> None:
        """Inicializa todos los chunkers del pipeline."""
        for chunker, _ in self._chunkers:
            chunker.setup(embedding_index=embedding_index, **kwargs)

    def teardown(self) -> None:
        """Limpia recursos de todos los chunkers."""
        for chunker, _ in self._chunkers:
            chunker.teardown()

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def run(self, text: str) -> ChunkingResult:
        """Ejecuta el pipeline completo sobre el texto dado.

        Returns:
            :class:`ChunkingResult` con fragmentos finales y metadata
            consolidada de todas las etapas.
        """
        # ── Primera etapa: texto completo ──────────────────────────────
        first_chunker, first_config = self._chunkers[0]
        result = first_chunker.chunk(text, first_config)

        stage_metadata: dict[str, Any] = {
            first_chunker.name: result.metadata,
        }

        # ── Etapas siguientes: refinar fragmento por fragmento ─────────
        for chunker, config in self._chunkers[1:]:
            refined_fragments: list[str] = []
            all_boundaries: list[ChunkBoundary] = []
            char_cursor = 0
            stage_dsp_signals: list[dict] = []  # DSP por fragmento

            for fragment in result.fragments:
                sub_result = chunker.chunk(fragment, config)

                # Acumular señales DSP por sub-fragmento
                if sub_result.has_dsp_signals:
                    stage_dsp_signals.append(sub_result.metadata)

                # Ajustar posiciones de boundaries (offset por char_cursor)
                for boundary in sub_result.boundaries:
                    all_boundaries.append(ChunkBoundary(
                        position=boundary.position + char_cursor,
                        confidence=boundary.confidence,
                        reason=boundary.reason,
                    ))

                for frag in sub_result.fragments:
                    refined_fragments.append(frag)
                    char_cursor += len(frag)

            # Construir metadata consolidada de esta etapa
            stage_meta: dict[str, Any] = {
                "chunker": chunker.name,
                "num_input_fragments": len(result.fragments),
                "num_output_fragments": len(refined_fragments),
            }

            # Consolidar señales DSP de todos los sub-fragmentos
            if stage_dsp_signals:
                stage_meta["per_fragment_dsp"] = stage_dsp_signals
                # Señal concatenada (útil para graficar todo el documento)
                all_sim = []
                all_smooth = []
                all_valleys = []
                offset = 0
                for dsp in stage_dsp_signals:
                    all_sim.extend(dsp.get("similarity_signal", []))
                    all_smooth.extend(dsp.get("smoothed_signal", []))
                    for v in dsp.get("valley_positions", []):
                        all_valleys.append(v + offset)
                    offset += dsp.get("num_units", 0)
                stage_meta["similarity_signal_concat"] = all_sim
                stage_meta["smoothed_signal_concat"] = all_smooth
                stage_meta["valley_positions_concat"] = all_valleys

            stage_metadata[chunker.name] = stage_meta

            result = ChunkingResult(
                fragments=refined_fragments,
                boundaries=all_boundaries,
                metadata=stage_metadata,
            )

        # Añadir summary al metadata final
        result.metadata["pipeline"] = self.name
        result.metadata["stages_order"] = [name for name, _ in self._stage_configs]
        result.metadata["total_chunks"] = len(result.fragments)

        return result

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Nombre legible del pipeline (para reportes y logs).

        Ejemplo: ``"structural → convolution(gaussian,5)"``
        """
        parts = []
        for chunker, config in self._chunkers:
            if chunker.name == "convolution":
                kt = config.get("kernel_type", "gaussian")
                ks = config.get("kernel_size", 5)
                parts.append(f"convolution({kt},{ks})")
            else:
                parts.append(chunker.name)
        return " → ".join(parts)

    @property
    def chunker_names(self) -> list[str]:
        """Nombres de los chunkers en el pipeline."""
        return [c.name for c, _ in self._chunkers]

    @property
    def configs(self) -> list[tuple[str, dict]]:
        """Configuraciones originales del pipeline."""
        return self._stage_configs

    # ------------------------------------------------------------------
    # Compatibilidad con NoteWatcher (interfaz de chunker simple)
    # ------------------------------------------------------------------

    def chunk(
        self,
        note_id: str,
        content: str,
        tags: list[str],
        priority: int,
    ) -> list[dict]:
        """Adapta el pipeline a la interfaz del NoteWatcher.

        El NoteWatcher espera un objeto con
        ``chunk(note_id, content, tags, priority) → list[dict]``.
        Este método delega a ``run()`` y convierte el resultado al
        formato de dict que NoteDB espera.

        Returns:
            Lista de dicts con ``id, title, content, order``.
        """
        result = self.run(content)
        chunks = []
        for i, fragment in enumerate(result.fragments):
            title = self._extract_title(fragment) or f"{note_id} parte {i + 1}"
            slug = self._slugify(title)
            chunks.append({
                "id": f"{note_id}::{slug}-{i}",
                "title": title,
                "content": fragment,
                "order": i,
                "_dsp_metadata": result.metadata,  # Para el ReportGenerator
            })
        return chunks

    @staticmethod
    def _extract_title(fragment: str) -> str | None:
        """Extrae el primer H1/H2 del fragmento."""
        import re
        for line in fragment.splitlines():
            m = re.match(r"^#{1,3}\s+(.+)$", line.strip())
            if m:
                return m.group(1).strip()
        # Primera oración como título
        first_line = fragment.strip().split("\n")[0].strip()
        return first_line[:60] if first_line else None

    @staticmethod
    def _slugify(title: str) -> str:
        import re
        slug = title.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug).strip("-")
        return slug or "chunk"
