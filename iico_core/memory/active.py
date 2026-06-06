"""
iico_core/memory/active.py
===========================
Registro de Skills (Memoria Activa).

Cada skill vive en un directorio propio:
    skills/
    ├── _registry.yaml       ← índice maestro: nombre → path relativo
    └── calculator/
        ├── meta.md          ← YAML frontmatter + descripción para el LLM
        └── run.py           ← implementación ejecutable

Formato de meta.md:
    ---
    name: calculator
    description: "Evalúa expresiones matemáticas simples."
    runtime: python
    tags: [matematicas, calculo, expresiones]
    input_schema:
      type: object
      properties:
        expression:
          type: string
          description: "Expresión matemática a evaluar (ej: '2 + 2 * 3')"
      required: [expression]
    output_schema:
      type: object
      properties:
        result:
          type: number
    ---
    # Calculator
    Evalúa expresiones matemáticas de forma segura...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import frontmatter
import yaml

from ..types import SkillDefinition


class SkillRegistry:
    """
    Gestiona el catálogo de skills disponibles para el agente.

    Responsabilidades:
    - Cargar las definiciones de skills desde disco al iniciar
    - Proveer las descripciones de tools al Harness (para el system prompt)
    - Resolver el nombre de una skill a su SkillDefinition (para el Bridge)
    """

    def __init__(self, skills_path: Path | str = "skills"):
        self.skills_path = Path(skills_path)
        self._skills: dict[str, SkillDefinition] = {}  # name → definición
        self.load()

    # ------------------------------------------------------------------
    # Carga desde disco
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carga el índice de skills. Soporta dos modos:
        1. Via _registry.yaml: lista explícita de skills habilitadas
        2. Discovery automático: escanea subdirectorios con meta.md
        """
        self._skills.clear()
        if not self.skills_path.exists():
            return

        registry_file = self.skills_path / "_registry.yaml"
        if registry_file.exists():
            self._load_from_registry(registry_file)
        else:
            self._discover_skills()

    def _load_from_registry(self, registry_file: Path) -> None:
        """Carga skills listadas en _registry.yaml."""
        try:
            with open(registry_file, encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[SkillRegistry] Error al leer _registry.yaml: {e}")
            return

        skills_list = registry.get("skills", [])
        for entry in skills_list:
            if isinstance(entry, str):
                skill_dir = self.skills_path / entry
            elif isinstance(entry, dict):
                skill_dir = self.skills_path / entry.get("path", entry.get("name", ""))
            else:
                continue

            skill = self._load_skill_dir(skill_dir)
            if skill:
                self._skills[skill.name] = skill

    def _discover_skills(self) -> None:
        """Escanea subdirectorios buscando meta.md automáticamente."""
        for skill_dir in self.skills_path.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                skill = self._load_skill_dir(skill_dir)
                if skill:
                    self._skills[skill.name] = skill

    def _load_skill_dir(self, skill_dir: Path) -> SkillDefinition | None:
        """Parsea el meta.md de un directorio de skill."""
        meta_path = skill_dir / "meta.md"
        if not meta_path.exists():
            return None

        try:
            post = frontmatter.load(str(meta_path))
            meta = post.metadata

            name = str(meta.get("name", skill_dir.name))
            description = str(meta.get("description", post.content.strip()[:200]))
            runtime = str(meta.get("runtime", "python"))
            tags_raw = meta.get("tags", [])
            tags = [str(t).lower() for t in tags_raw] if isinstance(tags_raw, list) else []

            input_schema = meta.get("input_schema", {
                "type": "object",
                "properties": {},
                "required": [],
            })
            output_schema = meta.get("output_schema", {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            })

            # Resolver el ejecutable según el runtime
            if runtime == "python":
                executable = skill_dir / "run.py"
            elif runtime == "shell":
                executable = skill_dir / "run.sh"
            else:
                executable = skill_dir / "run.py"  # fallback

            if not executable.exists():
                print(f"[SkillRegistry] Advertencia: {executable} no existe para skill '{name}'")

            return SkillDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                executable_path=executable,
                runtime=runtime,
                tags=tags,
            )

        except Exception as e:
            print(f"[SkillRegistry] Error al cargar skill en {skill_dir}: {e}")
            return None

    def reload(self) -> None:
        """Recarga el catálogo desde disco (útil en desarrollo)."""
        self.load()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get(self, skill_name: str) -> SkillDefinition | None:
        """Devuelve la definición de una skill por nombre."""
        return self._skills.get(skill_name)

    def get_tool_descriptions(self) -> list[dict]:
        """
        Genera la lista de descriptores de tools para el system prompt del LLM.
        Formato compatible con OpenAI tool calling / Ollama.
        """
        return [skill.to_tool_dict() for skill in self._skills.values()]

    def format_for_prompt(self) -> str:
        """Genera texto legible de las skills disponibles para el system prompt."""
        if not self._skills:
            return ""
        lines = ["## Skills disponibles\n"]
        for skill in self._skills.values():
            lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)

    def search_by_tags(self, query: str, max_results: int = 3) -> list[SkillDefinition]:
        """Búsqueda de skills por tags (para integración con el Splay Tree)."""
        normalized_query = self._normalize(query)
        query_words = set(re.findall(r"\b\w{2,}\b", normalized_query))
        if not query_words:
            return list(self._skills.values())[:max_results]

        scored: list[tuple[int, SkillDefinition]] = []
        for skill in self._skills.values():
            tag_set = set(self._normalize(t) for t in skill.tags)
            matches = len(query_words & tag_set)
            if matches > 0:
                scored.append((matches, skill))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:max_results]]

    @staticmethod
    def _normalize(text: str) -> str:
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        return self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[SkillDefinition]:
        return iter(self._skills.values())

    def __contains__(self, name: str) -> bool:
        return name in self._skills
