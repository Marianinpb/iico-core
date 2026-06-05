"""
iico_core/memory/passive.py
============================
Sistema de Memoria Pasiva: notas Markdown con frontmatter YAML.

Cada nota es un archivo .md en memory_store/ con esta estructura:

    ---
    id: protocolo-spi
    tags: [spi, comunicación, hardware]
    priority: 5          # 1-10, mayor = más relevante
    created: 2026-06-04
    ---
    # Protocolo SPI
    El SPI (Serial Peripheral Interface) es un bus de comunicación síncrono...

El Harness carga todas las notas al iniciar y las inyecta al system prompt
según relevancia (tags en Fase 1, embeddings en Fase 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import frontmatter  # python-frontmatter


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------

@dataclass
class PassiveNote:
    """Representa una nota de la Memoria Pasiva."""
    id: str
    tags: list[str]
    priority: int          # 1-10
    content: str           # Cuerpo Markdown (sin el frontmatter)
    source_path: Path | None = None

    def token_estimate(self) -> int:
        """Estimación rápida de tokens (1 token ≈ 4 caracteres en español)."""
        return len(self.content) // 4


# ---------------------------------------------------------------------------
# Memoria Pasiva
# ---------------------------------------------------------------------------

class PassiveMemory:
    """
    Gestor de notas Markdown+YAML.
    
    Fase 1: búsqueda determinista por tags (O(n), sin dependencias externas).
    Fase 2: se agregará búsqueda semántica via embeddings ONNX.
    """

    def __init__(self, memory_path: Path | str = "memory_store"):
        self.memory_path = Path(memory_path)
        self._notes: dict[str, PassiveNote] = {}   # id → nota
        self.load_all()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Parsea todas las notas .md del directorio memory_store/."""
        self._notes.clear()
        if not self.memory_path.exists():
            self.memory_path.mkdir(parents=True, exist_ok=True)
            return

        for path in self.memory_path.glob("*.md"):
            note = self._parse_note(path)
            if note:
                self._notes[note.id] = note

    def _parse_note(self, path: Path) -> PassiveNote | None:
        try:
            post = frontmatter.load(str(path))
            note_id   = str(post.metadata.get("id", path.stem))
            tags_raw  = post.metadata.get("tags", [])
            tags      = [str(t).lower() for t in tags_raw] if isinstance(tags_raw, list) else []
            priority  = int(post.metadata.get("priority", 5))
            content   = post.content.strip()
            return PassiveNote(
                id=note_id,
                tags=tags,
                priority=priority,
                content=content,
                source_path=path,
            )
        except Exception as e:
            print(f"[PassiveMemory] Error al parsear {path}: {e}")
            return None

    def reload(self) -> None:
        """Recarga las notas desde disco (útil si se crearon notas nuevas en sesión)."""
        self.load_all()

    # ------------------------------------------------------------------
    # Búsqueda (Fase 1: por tags)
    # ------------------------------------------------------------------

    def get_relevant(
        self,
        query: str,
        method: str = "tags",
        max_results: int = 5,
    ) -> list[PassiveNote]:
        """
        Devuelve notas relevantes para el query dado.

        Fase 1 — method='tags':
            Extrae palabras del query y las compara contra los tags de cada nota.
            Las notas con más coincidencias de tags aparecen primero.
            Empate resuelto por prioridad.
        """
        if not self._notes:
            return []

        if method == "tags":
            return self._search_by_tags(query, max_results)

        # Fase 2: "embeddings" — placeholder hasta que se implemente ONNX
        return []

    def _search_by_tags(self, query: str, max_results: int) -> list[PassiveNote]:
        # Tokenizar el query: palabras de ≥3 letras, en minúsculas
        query_words = set(
            w.lower() for w in re.findall(r"\b\w{3,}\b", query)
        )
        if not query_words:
            # Sin palabras útiles: devolver las notas de mayor prioridad
            return sorted(self._notes.values(), key=lambda n: -n.priority)[:max_results]

        scored: list[tuple[int, int, PassiveNote]] = []
        for note in self._notes.values():
            tag_set = set(note.tags)
            matches = len(query_words & tag_set)
            if matches > 0:
                scored.append((matches, note.priority, note))

        # Ordenar: más coincidencias primero, luego por prioridad
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [item[2] for item in scored[:max_results]]

    # ------------------------------------------------------------------
    # Presupuesto de tokens
    # ------------------------------------------------------------------

    def apply_token_budget(
        self,
        notes: list[PassiveNote],
        max_tokens: int,
    ) -> list[PassiveNote]:
        """
        Selecciona las notas que caben dentro del presupuesto de tokens.
        Ordena por prioridad (mayor primero) y corta cuando se excede el límite.
        """
        selected: list[PassiveNote] = []
        used = 0
        for note in sorted(notes, key=lambda n: -n.priority):
            est = note.token_estimate()
            if used + est <= max_tokens:
                selected.append(note)
                used += est
        return selected

    # ------------------------------------------------------------------
    # Creación de notas (el LLM puede crear notas via Harness)
    # ------------------------------------------------------------------

    def add_note(
        self,
        note_id: str,
        content: str,
        tags: list[str],
        priority: int = 5,
    ) -> PassiveNote:
        """Crea y persiste una nota nueva en memory_store/."""
        self.memory_path.mkdir(parents=True, exist_ok=True)
        path = self.memory_path / f"{note_id}.md"

        # Serializar con frontmatter
        post = frontmatter.Post(
            content,
            id=note_id,
            tags=tags,
            priority=priority,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        note = PassiveNote(
            id=note_id,
            tags=[t.lower() for t in tags],
            priority=priority,
            content=content,
            source_path=path,
        )
        self._notes[note_id] = note
        return note

    # ------------------------------------------------------------------
    # Formato para inyectar en el system prompt
    # ------------------------------------------------------------------

    def format_for_prompt(self, notes: list[PassiveNote]) -> str:
        """Convierte una lista de notas a texto para el system prompt."""
        if not notes:
            return ""
        parts = ["## Contexto relevante de tu memoria\n"]
        for note in notes:
            parts.append(f"### {note.id} [tags: {', '.join(note.tags)}]\n{note.content}\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    @property
    def notes(self) -> dict[str, PassiveNote]:
        return self._notes

    def __len__(self) -> int:
        return len(self._notes)

    def __iter__(self) -> Iterator[PassiveNote]:
        return iter(self._notes.values())
