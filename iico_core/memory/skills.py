from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import frontmatter


@dataclass
class SkillDefinition:
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    content: str = ""
    source_path: Path | None = None


class SkillLibrary:
    """
    Biblioteca de skills: flujos de trabajo en archivos .md.

    Soporta múltiples directorios: uno global (skills/) y uno local por proyecto
    (.iico/skills/). Skills locales sobrescriben globales si tienen el mismo nombre.

    Cada skill es un archivo Markdown con frontmatter YAML:
        ---
        name: analisis-datos-csv
        description: "Flujo para analizar datos CSV..."
        tags: [csv, datos, analisis]
        ---
        # Instrucciones de la skill
        ...

    A diferencia de las tools (ejecutables), las skills son solo instructivas:
    describen CÓMO secuenciar tools para tareas complejas.
    """

    def __init__(self, *paths: Path | str):
        self._paths: list[Path] = []
        self._skills: dict[str, SkillDefinition] = {}
        self._origins: dict[str, str] = {}  # skill_name -> etiqueta de origen
        for p in paths:
            self.add_path(p)

    def add_path(self, path: Path | str, label: str | None = None) -> None:
        """Agrega un directorio de skills. Skills locales sobrescriben globales."""
        p = Path(path)
        self._paths.append(p)
        origin_label = label or p.name
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            return
        for md_file in sorted(p.glob("*.md")):
            skill = self._parse_skill(md_file)
            if skill:
                self._skills[skill.name] = skill
                self._origins[skill.name] = origin_label

    def _parse_skill(self, path: Path) -> SkillDefinition | None:
        try:
            post = frontmatter.load(str(path))
            meta = post.metadata
            name = str(meta.get("name", path.stem))
            description = str(meta.get("description", ""))
            tags_raw = meta.get("tags", [])
            tags = [str(t).lower() for t in tags_raw] if isinstance(tags_raw, list) else []
            content = post.content.strip()
            return SkillDefinition(
                name=name,
                description=description,
                tags=tags,
                content=content,
                source_path=path,
            )
        except Exception as e:
            print(f"[SkillLibrary] Error al cargar skill {path}: {e}")
            return None

    def origin(self, name: str) -> str | None:
        """Etiqueta del directorio de origen de una skill."""
        return self._origins.get(name)

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def get_relevant(self, query: str, max_results: int = 3) -> list[SkillDefinition]:
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

    def format_for_prompt(self, max_skills: int = 5) -> str:
        if not self._skills:
            return ""
        lines = ["## Skills de flujo de trabajo disponibles\n"]
        for skill in list(self._skills.values())[:max_skills]:
            excerpt = skill.content[:200].replace("\n", " ").strip()
            lines.append(f"- **/{skill.name}**: {skill.description}")
            lines.append(f"  ```\n  {excerpt}...\n  ```")
        return "\n".join(lines)

    @staticmethod
    def _normalize(text: str) -> str:
        import unicodedata
        nfkd = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        return self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def __iter__(self) -> Iterator[SkillDefinition]:
        return iter(self._skills.values())

    def __contains__(self, name: str) -> bool:
        return name in self._skills
