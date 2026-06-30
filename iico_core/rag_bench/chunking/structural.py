"""
iico_core/rag_bench/chunking/structural.py
===========================================
Chunker determinista por estructura Markdown.

Corta en:
- Encabezados ``##`` y ``###``
- Bloques de código (```...```)
- Reglas horizontales (``---``, ``***``, ``___``)

Es el chunker base (baseline) sin ML ni embeddings.
Reutiliza internamente la lógica de
:class:`iico_core.memory.chunker.Chunker` para mantener consistencia
con el sistema de memoria existente.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .base import ChunkBoundary, ChunkingResult, ChunkingStrategy, register_chunker


# ---------------------------------------------------------------------------
# Sección interna (representación intermedia del parser)
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    title: str
    level: int   # 2=##, 3=###, 0=intro/código/separador
    body: str


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------

@register_chunker
class StructuralChunker(ChunkingStrategy):
    """Chunker determinista por encabezados Markdown.

    Algoritmo (misma lógica que el Chunker de memoria existente):

    1. Parsea el Markdown en secciones delimitadas por ``##``/``###``,
       bloques de código y reglas horizontales.
    2. Fusiona secciones ``###`` bajo su ``##`` padre si caben en el
       límite de tokens.
    3. Divide secciones que superan el límite por párrafos.
    4. Garantiza al menos 1 chunk por nota.

    Config keys:
        max_chunk_tokens (int): Máximo de tokens por chunk. Default: 512.
        min_chunk_chars (int): Mínimo de caracteres para crear un chunk.
                               Chunks más pequeños se fusionan con el anterior.
                               Default: 20.
    """

    name = "structural"
    description = "Corte determinista por encabezados Markdown (##/###), bloques de código y HR"

    def chunk(self, text: str, config: dict | None = None) -> ChunkingResult:
        cfg = config or {}
        max_tokens: int = cfg.get("max_chunk_tokens", 512)
        min_chars: int = cfg.get("min_chunk_chars", 20)

        sections = self._parse_markdown(text)
        fragments, boundaries = self._build_fragments(sections, max_tokens, min_chars)

        return ChunkingResult(
            fragments=fragments,
            boundaries=boundaries,
            metadata={
                "chunker": "structural",
                "max_chunk_tokens": max_tokens,
                "num_sections_parsed": len(sections),
                "num_chunks": len(fragments),
            },
        )

    # ------------------------------------------------------------------
    # Parser Markdown
    # ------------------------------------------------------------------

    def _parse_markdown(self, content: str) -> list[_Section]:
        """Parsea el Markdown en secciones (##/###, código, HR, intro)."""
        sections: list[_Section] = []
        stripped = content.strip()

        if not stripped:
            return [_Section("introduccion", 0, "")]

        current_title = "introduccion"
        current_level = 0
        current_lines: list[str] = []
        in_code_block = False
        code_lang = ""
        code_lines: list[str] = []

        for line in content.split("\n"):
            # ── Bloques de código ──
            if line.startswith("```"):
                if not in_code_block:
                    if current_lines:
                        sections.append(
                            _Section(current_title, current_level,
                                     "\n".join(current_lines))
                        )
                        current_lines = []
                    in_code_block = True
                    code_lang = line[3:].strip()
                    code_lines = []
                else:
                    in_code_block = False
                    code_title = f"codigo {code_lang}" if code_lang else "codigo"
                    sections.append(_Section(code_title, 0, "\n".join(code_lines)))
                    current_title = "continuacion"
                    current_level = 0
                continue

            if in_code_block:
                code_lines.append(line)
                continue

            # ── Reglas horizontales ──
            if line.strip() in ("---", "***", "___"):
                if current_lines:
                    sections.append(
                        _Section(current_title, current_level,
                                 "\n".join(current_lines))
                    )
                    current_lines = []
                current_title = "continuacion"
                current_level = 0
                continue

            # ── Encabezados ATX (## y ###) ──
            h_match = re.match(r"^(#{2,3})\s+(.+)$", line)
            if h_match:
                level = len(h_match.group(1))
                title = h_match.group(2).strip()
                if current_lines:
                    sections.append(
                        _Section(current_title, current_level,
                                 "\n".join(current_lines))
                    )
                    current_lines = []
                current_title = title
                current_level = level
                continue

            # ── Línea regular ──
            current_lines.append(line)

        # Última sección pendiente
        if current_lines:
            sections.append(
                _Section(current_title, current_level, "\n".join(current_lines))
            )

        return sections

    # ------------------------------------------------------------------
    # Construcción de fragmentos
    # ------------------------------------------------------------------

    def _build_fragments(
        self,
        sections: list[_Section],
        max_tokens: int,
        min_chars: int,
    ) -> tuple[list[str], list[ChunkBoundary]]:
        """Construye fragmentos finales: fusiona ### bajo ##, divide oversized."""
        fragments: list[str] = []
        boundaries: list[ChunkBoundary] = []
        char_cursor = 0
        i = 0

        while i < len(sections):
            section = sections[i]

            # ── Fusión de ### bajo ## ──
            if section.level == 2:
                merged_body = section.body
                j = i + 1
                while j < len(sections) and sections[j].level == 3:
                    merged_body += (
                        "\n\n### " + sections[j].title + "\n" + sections[j].body
                    )
                    j += 1

                body_to_use = merged_body if j > i + 1 else section.body

                if not body_to_use.strip():
                    i = j if j > i + 1 else i + 1
                    continue

                sub_fragments = self._maybe_split(body_to_use, max_tokens)
                for frag in sub_fragments:
                    if len(frag) >= min_chars:
                        if fragments:  # No boundary antes del primer chunk
                            boundaries.append(ChunkBoundary(
                                position=char_cursor,
                                confidence=0.9,
                                reason=f"header_h{section.level}({section.title!r})",
                            ))
                        fragments.append(frag)
                        char_cursor += len(frag)
                i = j if j > i + 1 else i + 1
                continue

            # ── Sección regular (nivel 0 o 3) ──
            if not section.body.strip():
                i += 1
                continue

            sub_fragments = self._maybe_split(section.body, max_tokens)
            for frag in sub_fragments:
                if len(frag) >= min_chars:
                    if fragments:
                        reason = (
                            f"header_h{section.level}({section.title!r})"
                            if section.level > 0
                            else "structural_separator"
                        )
                        boundaries.append(ChunkBoundary(
                            position=char_cursor,
                            confidence=0.8 if section.level > 0 else 0.5,
                            reason=reason,
                        ))
                    fragments.append(frag)
                    char_cursor += len(frag)
            i += 1

        # Garantizar al menos 1 fragmento
        if not fragments:
            fragments = [sections[0].body if sections else ""]
            boundaries = []

        return fragments, boundaries

    def _maybe_split(self, body: str, max_tokens: int) -> list[str]:
        """Divide un cuerpo por párrafos si supera max_tokens."""
        if self._estimate_tokens(body) <= max_tokens:
            return [body.strip()]
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if not paragraphs:
            return [body.strip()]

        result: list[str] = []
        batch: list[str] = []
        batch_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if batch and batch_tokens + para_tokens > max_tokens:
                result.append("\n\n".join(batch))
                batch = []
                batch_tokens = 0
            batch.append(para)
            batch_tokens += para_tokens

        if batch:
            result.append("\n\n".join(batch))

        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimación rápida: 1 token ≈ 4 caracteres."""
        return len(text) // 4
