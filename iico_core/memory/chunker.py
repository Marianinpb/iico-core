"""
iico_core/memory/chunker.py
============================
Chunker estructural de Markdown: divide una PassiveNote en Chunks
basándose en la estructura de encabezados, bloques de código y
reglas horizontales.

El chunking es determinista (sin ML) y sigue únicamente la sintaxis
Markdown: encabezados ATX nivel 2+, bloques de código con backticks
triples, y reglas horizontales (---, ***, ___).

Fase 4 — Memoria Particionada.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import Chunk

if TYPE_CHECKING:
    from ..index.embedding import EmbeddingIndex
    from .passive import PassiveNote


# ---------------------------------------------------------------------------
# Sección intermedia (representación interna del parser)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Section:
    title: str
    level: int   # 2=##, 3=###, 0=intro/código/separador
    body: str


# ---------------------------------------------------------------------------
# SemanticSplitter: división semántica con ventanas deslizantes + cosine similarity
# ---------------------------------------------------------------------------

class SemanticSplitter:
    """Divide texto en límites semánticos usando ventanas deslizantes + cosine similarity.

    Usa un EmbeddingIndex ya cargado para vectorizar ventanas y detectar puntos
    donde el tema cambia (cosine similarity cae bajo un umbral).

    El splitter es opcional en el pipeline de Chunker: si no se provee, el Chunker
    divide únicamente por párrafos. Si se provee, refina fragmentos que siguen
    excediendo el límite de tokens incluso tras la división por párrafos.
    """

    def __init__(
        self,
        embedding_index: "EmbeddingIndex",
        window_tokens: int = 200,
        stride_tokens: int = 50,
    ) -> None:
        self._index = embedding_index
        self.window_tokens = window_tokens
        self.stride_tokens = stride_tokens

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def split(self, text: str, threshold: float = 0.5) -> list[str]:
        """Divide el texto en los puntos de quiebre semántico.

        Retorna lista de fragmentos. Si no se detecta ningún quiebre,
        retorna [text] (el texto completo como un solo fragmento).

        Args:
            text: texto a dividir semánticamente
            threshold: umbral mínimo de cosine similarity para considerar
                       dos ventanas como del mismo tema (0.0 - 1.0)
        """
        window_chars = self.window_tokens * 4

        # Texto muy corto: no tiene sentido dividir
        if len(text) < window_chars:
            return [text]

        windows = self._sliding_windows(text)
        break_points = self._compute_break_points(windows, threshold)

        if not break_points:
            return [text]

        # Dividir el texto en los puntos de quiebre
        fragments: list[str] = []
        prev = 0
        for bp in break_points:
            fragment = text[prev:bp].strip()
            if fragment:
                fragments.append(fragment)
            prev = bp
        # Último fragmento tras el último punto de quiebre
        last = text[prev:].strip()
        if last:
            fragments.append(last)

        return fragments if fragments else [text]

    # ------------------------------------------------------------------
    # Ventanas deslizantes
    # ------------------------------------------------------------------

    def _sliding_windows(self, text: str) -> list[str]:
        """Genera ventanas deslizantes del texto.

        Cada ventana ≈ window_tokens * 4 caracteres.
        El stride (paso) ≈ stride_tokens * 4 caracteres.
        La última ventana puede ser más corta si el texto no alcanza.
        """
        window_chars = self.window_tokens * 4
        stride_chars = self.stride_tokens * 4

        windows: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + window_chars, len(text))
            windows.append(text[start:end])
            start += stride_chars

        return windows

    # ------------------------------------------------------------------
    # Puntos de quiebre semántico
    # ------------------------------------------------------------------

    def _compute_break_points(
        self, windows: list[str], threshold: float
    ) -> list[int]:
        """Vectoriza cada ventana, calcula cosine similarity consecutiva.

        Retorna posiciones de caracteres donde la similarity entre ventanas
        consecutivas cae por debajo del threshold (puntos de quiebre).
        """
        from ..index.embedding import cosine_similarity

        if len(windows) < 2:
            return []

        stride_chars = self.stride_tokens * 4

        # Vectorizar todas las ventanas
        vectors = [self._index.vectorize(w) for w in windows]

        # Calcular cosine similarity entre ventanas consecutivas
        break_points: list[int] = []
        for i in range(len(vectors) - 1):
            sim = cosine_similarity(vectors[i], vectors[i + 1])
            if sim < threshold:
                # Punto de quiebre: inicio de la ventana i+1
                bp = (i + 1) * stride_chars
                break_points.append(bp)

        return break_points


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Divide una PassiveNote en Chunks basándose en estructura Markdown.

    Parser determinista en una pasada. Sin dependencias externas de Markdown ni ML.
    Opcionalmente, usa un SemanticSplitter para refinar secciones sobredimensionadas
    que ni la división por párrafos logra acotar.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        semantic_splitter: "SemanticSplitter | None" = None,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self._semantic = semantic_splitter

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def chunk_note(self, note: PassiveNote) -> list[Chunk]:
        """Divide una nota en chunks: parse → fusionar ### bajo ## → dividir oversized → construir Chunks."""
        sections = self._parse_markdown(note.content)
        chunks = self._build_chunks(note, sections)
        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimación rápida de tokens (1 token ≈ 4 caracteres en español)."""
        return len(text) // 4

    def _generate_chunk_id(self, parent_note_id: str, title: str) -> str:
        """Genera un ID de chunk a partir del ID de la nota padre y el título.

        Formato: ``parent_note_id::slug-del-titulo``
        Ejemplo: ``arquitectura_harness::splay-rotations``
        """
        slug = self._slugify(title)
        return f"{parent_note_id}::{slug}"

    @staticmethod
    def _slugify(title: str) -> str:
        """Convierte un título en un slug seguro para IDs de archivo.

        - Minúsculas
        - Solo caracteres alfanuméricos, espacios y guiones
        - Espacios → guiones
        - Guiones múltiples colapsados
        - Guiones iniciales/finales eliminados

        Ejemplo: ``Rotaciones Splay (zig, zig-zag)`` → ``rotaciones-splay-zig-zig-zag``
        """
        slug = title.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        slug = slug.strip("-")
        return slug if slug else "sin-titulo"

    # ------------------------------------------------------------------
    # Parser de Markdown
    # ------------------------------------------------------------------

    def _parse_markdown(self, content: str) -> list[_Section]:
        """Parsea Markdown en secciones: ##/### headers, bloques ```, HR (---/***/___).

        Contenido antes del primer header → sección "introduccion".
        """
        sections: list[_Section] = []

        # Caso borde: contenido vacío
        stripped = content.strip()
        if not stripped:
            return [_Section("introduccion", 0, "")]

        current_title = "introduccion"
        current_level = 0
        current_lines: list[str] = []

        in_code_block = False
        code_block_lang = ""
        code_block_lines: list[str] = []

        for line in content.split("\n"):
            # ── Bloques de código ──
            if line.startswith("```"):
                if not in_code_block:
                    # Guardar sección actual antes del bloque de código
                    if current_lines:
                        sections.append(
                            _Section(current_title, current_level, "\n".join(current_lines))
                        )
                        current_lines = []
                    in_code_block = True
                    code_block_lang = line[3:].strip()
                    code_block_lines = []
                else:
                    # Cerrar bloque de código
                    in_code_block = False
                    code_title = f"codigo {code_block_lang}" if code_block_lang else "codigo"
                    sections.append(_Section(code_title, 0, "\n".join(code_block_lines)))
                    # Reiniciar contexto de sección tras bloque de código
                    current_title = "continuacion"
                    current_level = 0
                continue

            if in_code_block:
                code_block_lines.append(line)
                continue

            # ── Reglas horizontales ──
            stripped_line = line.strip()
            if stripped_line in ("---", "***", "___"):
                if current_lines:
                    sections.append(
                        _Section(current_title, current_level, "\n".join(current_lines))
                    )
                    current_lines = []
                # Tras separador, reiniciar contexto para no arrastrar el título anterior
                current_title = "continuacion"
                current_level = 0
                continue

            # ── Encabezados ATX (## y ###) ──
            header_match = re.match(r"^(#{2,3})\s+(.+)$", line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                if current_lines:
                    sections.append(
                        _Section(current_title, current_level, "\n".join(current_lines))
                    )
                    current_lines = []

                current_title = title
                current_level = level
                continue

            # ── Línea regular ──
            current_lines.append(line)

        # ── Última sección ──
        if current_lines:
            sections.append(
                _Section(current_title, current_level, "\n".join(current_lines))
            )

        return sections

    # ------------------------------------------------------------------
    # Construcción de Chunks
    # ------------------------------------------------------------------

    def _build_chunks(self, note: PassiveNote, sections: list[_Section]) -> list[Chunk]:
        """Construye Chunks: fusiona ### bajo ## si caben, divide oversized por párrafos."""
        chunks: list[Chunk] = []
        order_counter = 0
        i = 0

        while i < len(sections):
            section = sections[i]

            # ── Fusión de ### bajo ## ──
            if section.level == 2:
                # Calcular cuerpo fusionado: el ## + todos sus ### hijos
                merged_body = section.body
                j = i + 1
                while j < len(sections) and sections[j].level == 3:
                    merged_body += (
                        "\n\n### " + sections[j].title + "\n" + sections[j].body
                    )
                    j += 1

                # ¿El cuerpo fusionado tiene contenido real?
                if not merged_body.strip():
                    i = j if j > i + 1 else i + 1
                    continue

                if j > i + 1 and self._estimate_tokens(merged_body) <= self.max_chunk_tokens:
                    # Fusionar: un solo chunk para el ## con todos sus ###
                    chunks.append(
                        self._make_chunk(note, section.title, merged_body, order_counter)
                    )
                    order_counter += 1
                    i = j
                    continue

                # No se fusiona (o no hay sub-secciones): chunk solo para el ##
                if section.body.strip():
                    if self._estimate_tokens(section.body) <= self.max_chunk_tokens:
                        chunks.append(
                            self._make_chunk(
                                note, section.title, section.body, order_counter
                            )
                        )
                    else:
                        sub = self._split_oversized(
                            note, section.title, section.body, order_counter
                        )
                        chunks.extend(sub)
                        order_counter += len(sub)
                        i += 1
                        continue
                    order_counter += 1
                i += 1
                continue

            # ── Sección regular (nivel 3, 0) ──
            if not section.body.strip():
                i += 1
                continue

            if self._estimate_tokens(section.body) <= self.max_chunk_tokens:
                chunks.append(
                    self._make_chunk(note, section.title, section.body, order_counter)
                )
            else:
                sub = self._split_oversized(
                    note, section.title, section.body, order_counter
                )
                chunks.extend(sub)
                order_counter += len(sub)
                i += 1
                continue
            order_counter += 1
            i += 1

        # ── Garantizar al menos un chunk (nota completamente vacía) ──
        if not chunks:
            chunks.append(self._make_chunk(note, "introduccion", "", 0))

        return chunks

    # ------------------------------------------------------------------
    # División de secciones sobredimensionadas
    # ------------------------------------------------------------------

    def _split_oversized(
        self, note: PassiveNote, title: str, body: str, start_order: int
    ) -> list[Chunk]:
        """Divide sección grande por párrafos, con refinamiento semántico opcional.

        Pipeline de dos pasadas:
        1. División por párrafos (\\n\\n)
        2. Refinamiento semántico en fragmentos que siguen excediendo el límite
           (solo si self._semantic está disponible)
        """
        # ── Pasada 1: dividir por párrafos ──
        fragments: list[str] = body.split("\n\n")

        # ── Pasada 2: refinamiento semántico en fragmentos sobredimensionados ──
        if self._semantic is not None:
            refined: list[str] = []
            for frag in fragments:
                if self._estimate_tokens(frag) > self.max_chunk_tokens:
                    semantic_frags = self._semantic.split(frag)
                    refined.extend(semantic_frags)
                else:
                    refined.append(frag)
            fragments = refined

        # ── Agrupar fragmentos en chunks respetando max_chunk_tokens ──
        chunks: list[Chunk] = []
        order = start_order
        part = 1
        batch: list[str] = []
        batch_tokens = 0

        for frag in fragments:
            frag_tokens = self._estimate_tokens(frag)

            if batch and batch_tokens + frag_tokens > self.max_chunk_tokens:
                part_title = f"{title} (parte {part})"
                chunks.append(
                    self._make_chunk(note, part_title, "\n\n".join(batch), order)
                )
                order += 1
                part += 1
                batch = []
                batch_tokens = 0

            batch.append(frag)
            batch_tokens += frag_tokens

        # Último batch
        if batch:
            part_title = f"{title} (parte {part})" if part > 1 else title
            chunks.append(
                self._make_chunk(note, part_title, "\n\n".join(batch), order)
            )

        return chunks

    # ------------------------------------------------------------------
    # Construcción de un Chunk individual
    # ------------------------------------------------------------------

    def _make_chunk(
        self, note: PassiveNote, title: str, content: str, order: int
    ) -> Chunk:
        """Construye un Chunk con todos los metadatos heredados del parent."""
        content_stripped = content.strip()
        return Chunk(
            id=self._generate_chunk_id(note.id, title),
            parent_note_id=note.id,
            title=title,
            content=content_stripped,
            tags=list(note.tags),
            priority=note.priority,
            order=order,
            content_hash=hashlib.sha256(
                content_stripped.encode("utf-8")
            ).hexdigest(),
        )
