"""
iico_core/rag_bench/reports/generator.py
==========================================
Generador de reportes: Markdown, CSV, JSON + exportación de señales DSP.

Genera 3 tipos de salida para cada benchmark run:

1. **Markdown** (``report.md``): Tabla comparativa de métricas, listado
   de queries con resultados individuales, gráfica ASCII de señales DSP.

2. **CSV** (``metrics.csv``): Una fila por query, todas las métricas.
   Listo para importar en pandas/Excel para el análisis de la tesis.

3. **JSON** (``dsp_signals.json``): Señales DSP del ConvolutionChunker
   (similarity_signal, smoothed_signal, valley_positions) para graficar
   correlogramas en notebooks.

Ejemplo de uso::

    from iico_core.rag_bench.reports import ReportGenerator

    gen = ReportGenerator(output_dir="benchmarks/results")
    gen.generate(run)                        # un solo run
    gen.compare([run1, run2, run3])          # tabla comparativa de múltiples runs
"""

from __future__ import annotations

import csv
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import BenchmarkRun, QueryResult


class ReportGenerator:
    """Genera reportes de benchmark en múltiples formatos.

    Args:
        output_dir: Directorio raíz donde se guardan los reportes.
                    Se crea automáticamente si no existe.
    """

    def __init__(self, output_dir: str | Path = "benchmarks/results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(self, run: BenchmarkRun) -> dict[str, Path]:
        """Genera todos los reportes para un run individual.

        Args:
            run: :class:`BenchmarkRun` con resultados completos.

        Returns:
            Dict ``{"markdown": path, "csv": path, "json": path}`` con
            las rutas de los archivos generados.
        """
        run_dir = self.output_dir / run.config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "markdown": self._write_markdown(run, run_dir),
            "csv":      self._write_csv(run, run_dir),
            "json":     self._write_json(run, run_dir),
        }

        # Exportar señales DSP si hay ConvolutionChunker en el pipeline
        dsp_path = self._write_dsp_signals(run, run_dir)
        if dsp_path:
            paths["dsp"] = dsp_path

        return paths

    def compare(self, runs: list[BenchmarkRun]) -> Path:
        """Genera una tabla comparativa de múltiples runs en un solo Markdown.

        Args:
            runs: Lista de :class:`BenchmarkRun` a comparar.

        Returns:
            Ruta al archivo ``comparison.md`` generado.
        """
        if not runs:
            return self.output_dir / "comparison.md"

        path = self.output_dir / "comparison.md"
        content = self._build_comparison_markdown(runs)
        path.write_text(content, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _write_markdown(self, run: BenchmarkRun, run_dir: Path) -> Path:
        path = run_dir / "report.md"
        content = self._build_run_markdown(run)
        path.write_text(content, encoding="utf-8")
        return path

    def _build_run_markdown(self, run: BenchmarkRun) -> str:
        cfg = run.config
        agg = run.aggregated
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines = [
            f"# Benchmark Report — {cfg.run_id}",
            f"\n> Generado: {ts}",
            f"\n## Configuración",
            f"",
            f"| Campo | Valor |",
            f"|-------|-------|",
            f"| Run ID | `{cfg.run_id}` |",
            f"| Chunking | `{' → '.join(n for n, _ in cfg.chunking_pipeline)}` |",
            f"| Retrieval | `{cfg.retrieval_strategy}` |",
            f"| Top-K | {cfg.top_k} |",
            f"| RAGAS | {'✅ habilitado' if cfg.enable_ragas else '❌ deshabilitado'} |",
            f"| Queries evaluadas | {agg.num_queries} |",
            f"| Tiempo total | {agg.total_elapsed_ms:.0f} ms |",
            f"",

            f"### Performance",
            f"",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Latencia promedio | {agg.avg_latency_ms:.2f} ms |",
            f"| Tokens de contexto promedio | {agg.avg_context_tokens:.0f} |",
        ]

        if cfg.enable_ragas:
            lines += [
                f"| RAGAS Score promedio | {agg.avg_ragas_score:.4f} |",
                f"| RAGAS P (Context Precision) | {agg.avg_ragas_context_precision:.4f} |",
                f"| RAGAS R (Context Recall) | {agg.avg_ragas_context_recall:.4f} |",
                f"| E_tok promedio | **{agg.avg_e_tok:.1f}** |",
                f"| RAGAS cache hit rate | {agg.ragas_cache_hit_rate:.2%} |",
            ]

        if agg.splay_hit_rate > 0:
            lines.append(f"| Splay cache hit rate | {agg.splay_hit_rate:.2%} |")

        # Tabla por query
        lines += [f"", f"## Resultados por Query", f""]
        if cfg.enable_ragas:
            lines.append(f"| ID | Query | Lat(ms) | Tokens | RAGAS | RAGAS P | RAGAS R |")
            lines.append(f"|----|-------|---------|--------|-------|---------|---------|")
        else:
            lines.append(f"| ID | Query | Lat(ms) | Tokens |")
            lines.append(f"|----|-------|---------|--------|")

        for qr in run.query_results:
            q_trunc = qr.query[:50] + "..." if len(qr.query) > 50 else qr.query
            row = f"| `{qr.query_id}` | {q_trunc} | {qr.perf.retrieval_latency_ms:.1f} | {qr.perf.total_context_tokens} |"
            if cfg.enable_ragas:
                row += f" {qr.perf.ragas_score:.4f} | {qr.ragas.context_precision:.4f} | {qr.ragas.context_recall:.4f} |"
            lines.append(row)

        # Sección de errores
        if run.errors:
            lines += [
                f"",
                f"## Errores ({len(run.errors)})",
                f"",
            ]
            for err in run.errors:
                lines.append(f"- {err}")

        lines += [
            f"",
            f"---",
            f"### ⚡ Métricas de Desempeño y Costo",
            f"- **Latencia (ms)**: Tiempo en ms para buscar información. Fundamental en sistemas embebidos.",
            f"- **Tokens**: Texto inyectado al LLM. Define el consumo de memoria VRAM.",
            f"- **RAGAS Score (0 a 1)**: Evaluación automatizada (LLM Juez) que verifica si el texto responde la pregunta.",
            f"- **E_tok ($E_{{tok}}$)**: Eficiencia ($Tokens / RAGAS$). **MENOR es MEJOR**. Evalúa la memoria invertida por punto de calidad.",
            f"- **RAGAS P (Context Precision)**: Evalúa si los fragmentos relevantes están bien posicionados. (LLM Evaluated).",
            f"- **RAGAS R (Context Recall)**: Evalúa si el contexto recuperado logra alinear toda la respuesta esperada. (LLM Evaluated).",
            f"",
            f"### 🧠 Estrategias Evaluadas (Aportes de Tesis)",
            f"#### 1. Fases de Chunking (Segmentación)",
            f"- **document**: No divide el texto. La nota completa es un solo chunk. Sirve como línea base del peor rendimiento (satura el contexto).",
            f"- **naive**: Corta el texto estáticamente por cantidad de tokens (ej. 200, 500). El método más popular pero ignorante del contenido.",
            f"- **structural**: Divide estáticamente respetando los encabezados Markdown y párrafos. Es el método más lógico para documentos estructurados.",
            f"- **semantic**: Corta midiendo la similitud del coseno entre oraciones, creando un nuevo chunk cuando detecta un cambio brusco de tema.",
            f"- **[cualquiera] → convolution** *(Aporte)*: Toma los fragmentos de la etapa anterior y aplica filtros de procesamiento de señales (convolución) para fusionar dinámicamente aquellos que comparten contexto, mejorando la cohesión.",
            f"",
            f"#### 2. Fases de Recuperación (Retrieval)",
            f"- **embeddings**: RAG tradicional. Búsqueda vectorial exhaustiva por distancia coseno. Precisión alta, latencia y costo computacional alto.",
            f"- **splay** *(Aporte)*: Caché adaptativa Splay Tree. Reorganiza accesos recientes para retornar hits en ~0ms sin pasar por inferencia ONNX, optimizando drásticamente la latencia en hardware limitado.",
        ]

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Tabla comparativa (múltiples runs)
    # ------------------------------------------------------------------

    def _build_comparison_markdown(self, runs: list[BenchmarkRun]) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        has_ragas = any(r.config.enable_ragas for r in runs)

        lines = [
            f"# Comparación de Estrategias RAG",
            f"\n> Generado: {ts}  |  {len(runs)} runs",
            f"",
            f"## Tabla Comparativa",
            f"",
        ]

        header = "| Run ID | Chunking | Retrieval | Chunk(ms) | Ret(ms) | Tokens |"
        sep = "|--------|----------|-----------|-----------|---------|--------|"
        
        if has_ragas:
            header += " RAGAS | RAGAS P | RAGAS R | E_tok |"
            sep += "-------|---------|---------|-------|"
        else:
            header += " |"
            sep += "|"

        lines += [header, sep]

        # Fila por run
        best_e_tok = min((r.aggregated.avg_e_tok for r in runs if r.config.enable_ragas), default=None)
        best_chunk_lat = min(r.aggregated.chunking_latency_ms for r in runs)
        best_lat = min(r.aggregated.avg_latency_ms for r in runs)
        best_tokens = min(r.aggregated.avg_context_tokens for r in runs)
        best_ragas = max((r.aggregated.avg_ragas_score for r in runs if r.config.enable_ragas), default=None)
        best_ragas_p = max((r.aggregated.avg_ragas_context_precision for r in runs if r.config.enable_ragas), default=None)
        best_ragas_r = max((r.aggregated.avg_ragas_context_recall for r in runs if r.config.enable_ragas), default=None)

        for run in runs:
            agg = run.aggregated
            cfg = run.config
            chunking = " → ".join(n for n, _ in cfg.chunking_pipeline)

            chunk_str = f"**{agg.chunking_latency_ms:.1f}**" if agg.chunking_latency_ms == best_chunk_lat else f"{agg.chunking_latency_ms:.1f}"
            lat_str = f"**{agg.avg_latency_ms:.1f}**" if agg.avg_latency_ms == best_lat else f"{agg.avg_latency_ms:.1f}"
            tok_str = f"**{agg.avg_context_tokens:.0f}**" if agg.avg_context_tokens == best_tokens else f"{agg.avg_context_tokens:.0f}"

            row = (
                f"| `{cfg.run_id}` | {chunking} | {cfg.retrieval_strategy} | "
                f"{chunk_str} | {lat_str} | {tok_str}"
            )

            if has_ragas and cfg.enable_ragas:
                ragas_str = f"**{agg.avg_ragas_score:.4f}**" if best_ragas and agg.avg_ragas_score == best_ragas else f"{agg.avg_ragas_score:.4f}"
                ragas_p_str = f"**{agg.avg_ragas_context_precision:.4f}**" if best_ragas_p and agg.avg_ragas_context_precision == best_ragas_p else f"{agg.avg_ragas_context_precision:.4f}"
                ragas_r_str = f"**{agg.avg_ragas_context_recall:.4f}**" if best_ragas_r and agg.avg_ragas_context_recall == best_ragas_r else f"{agg.avg_ragas_context_recall:.4f}"
                e_tok_str = f"**{agg.avg_e_tok:.1f}**" if best_e_tok and agg.avg_e_tok == best_e_tok else f"{agg.avg_e_tok:.1f}"
                row += f" | {ragas_str} | {ragas_p_str} | {ragas_r_str} | {e_tok_str} |"
            elif has_ragas:
                row += " | - | - | - | - |"
            else:
                row += " |"

            lines.append(row)

        # Notas interpretativas
        lines += [
            f"",
            f"> **Nota**: Los valores en **negrita** indican el mejor rendimiento en esa métrica.",
            f"> E_tok menor = más eficiente (menos tokens por unidad de calidad RAGAS).",
            f"",
            f"---",
            f"### ⚡ Métricas de Desempeño y Costo",
            f"- **Chunk(ms)**: Tiempo en ms para fragmentar la base de conocimiento (overhead de ingesta).",
            f"- **Ret(ms)**: Tiempo en ms para recuperar información durante una query. Fundamental en sistemas embebidos.",
            f"- **Tokens**: Texto inyectado al LLM. Define el consumo de memoria VRAM.",
            f"- **RAGAS Score (0 a 1)**: Evaluación automatizada (LLM Juez) que verifica si el texto responde la pregunta.",
            f"- **E_tok ($E_{{tok}}$)**: Eficiencia ($Tokens / RAGAS$). **MENOR es MEJOR**. Evalúa la memoria invertida por punto de calidad.",
            f"- **RAGAS P (Context Precision)**: Evalúa si los fragmentos relevantes están bien posicionados. (LLM Evaluated).",
            f"- **RAGAS R (Context Recall)**: Evalúa si el contexto recuperado logra alinear toda la respuesta esperada. (LLM Evaluated).",
            f"",
            f"### 🧠 Estrategias Evaluadas (Aportes de Tesis)",
            f"#### 1. Fases de Chunking (Segmentación)",
            f"- **document**: No divide el texto. La nota completa es un solo chunk. Sirve como línea base del peor rendimiento (satura el contexto).",
            f"- **naive**: Corta el texto estáticamente por cantidad de tokens (ej. 200, 500). El método más popular pero ignorante del contenido.",
            f"- **structural**: Divide estáticamente respetando los encabezados Markdown y párrafos. Es el método más lógico para documentos estructurados.",
            f"- **semantic**: Corta midiendo la similitud del coseno entre oraciones, creando un nuevo chunk cuando detecta un cambio brusco de tema.",
            f"- **[cualquiera] → convolution** *(Aporte)*: Toma los fragmentos de la etapa anterior y aplica filtros de procesamiento de señales (convolución) para fusionar dinámicamente aquellos que comparten contexto, mejorando la cohesión.",
            f"",
            f"#### 2. Fases de Recuperación (Retrieval)",
            f"- **embeddings**: RAG tradicional. Búsqueda vectorial exhaustiva por distancia coseno. Precisión alta, latencia y costo computacional alto.",
            f"- **splay** *(Aporte)*: Caché adaptativa Splay Tree. Reorganiza accesos recientes para retornar hits en ~0ms sin pasar por inferencia ONNX, optimizando drásticamente la latencia en hardware limitado."
        ]

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _write_csv(self, run: BenchmarkRun, run_dir: Path) -> Path:
        path = run_dir / "metrics.csv"
        has_ragas = run.config.enable_ragas

        fieldnames = [
            "query_id", "query",
            "latency_ms", "context_tokens", "cache_hit",
        ]
        if has_ragas:
            fieldnames += ["ragas_score", "e_tok", "ragas_faithfulness",
                           "ragas_answer_relevancy", "ragas_from_cache"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qr in run.query_results:
                row = {
                    "query_id": qr.query_id,
                    "query": qr.query,
                    "latency_ms": round(qr.perf.retrieval_latency_ms, 2),
                    "context_tokens": qr.perf.total_context_tokens,
                    "cache_hit": qr.perf.cache_hit,
                }
                if has_ragas:
                    row["ragas_score"] = round(qr.perf.ragas_score, 4)
                    row["e_tok"] = round(qr.perf.e_tok, 1)
                    if qr.ragas:
                        row["ragas_faithfulness"] = round(qr.ragas.faithfulness, 4)
                        row["ragas_answer_relevancy"] = round(qr.ragas.answer_relevancy, 4)
                        row["ragas_from_cache"] = qr.ragas.from_cache
                    else:
                        row.update({"ragas_faithfulness": 0, "ragas_answer_relevancy": 0,
                                    "ragas_from_cache": False})
                writer.writerow(row)

        return path

    # ------------------------------------------------------------------
    # JSON + señales DSP
    # ------------------------------------------------------------------

    def _write_json(self, run: BenchmarkRun, run_dir: Path) -> Path:
        """Exporta todo el run a JSON (sin señales DSP pesadas)."""
        path = run_dir / "run_summary.json"
        data = {
            "run_id": run.config.run_id,
            "chunking": [n for n, _ in run.config.chunking_pipeline],
            "retrieval_strategy": run.config.retrieval_strategy,
            "top_k": run.config.top_k,
            "enable_ragas": run.config.enable_ragas,
            "aggregated": {
                "num_queries": run.aggregated.num_queries,
                "avg_latency_ms": round(run.aggregated.avg_latency_ms, 2),
                "avg_context_tokens": round(run.aggregated.avg_context_tokens, 1),
                "avg_ragas_score": round(run.aggregated.avg_ragas_score, 4),
                "avg_ragas_context_precision": round(run.aggregated.avg_ragas_context_precision, 4),
                "avg_ragas_context_recall": round(run.aggregated.avg_ragas_context_recall, 4),
                "avg_e_tok": round(run.aggregated.avg_e_tok, 2),
                "ragas_cache_hit_rate": round(run.aggregated.ragas_cache_hit_rate, 4),
                "splay_hit_rate": round(run.aggregated.splay_hit_rate, 4),
                "total_elapsed_ms": round(run.aggregated.total_elapsed_ms, 1),
            },
            "errors": run.errors,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_dsp_signals(self, run: BenchmarkRun, run_dir: Path) -> Path | None:
        """Exporta señales DSP del ConvolutionChunker a JSON separado.

        Solo se genera si algún QueryResult tiene metadata DSP.
        Útil para cargar en notebooks y generar correlogramas de la tesis.
        """
        # Recolectar señales DSP de todas las queries
        dsp_data: list[dict] = []
        for qr in run.query_results:
            meta = qr.dsp_metadata
            if meta and ("similarity_signal" in meta or "stages" in meta):
                dsp_data.append({
                    "query_id": qr.query_id,
                    "query": qr.query,
                    "dsp": meta,
                })

        if not dsp_data:
            return None

        path = run_dir / "dsp_signals.json"
        path.write_text(
            json.dumps({"run_id": run.config.run_id, "signals": dsp_data},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path
