"""
iico_core/memory/active.py
===========================
Registro de Tools (Memoria Activa).

Cada tool vive en un directorio propio:
    tools/
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

from ..types import ToolDefinition


class ToolRegistry:
    """
    Gestiona el catálogo de tools disponibles para el agente.

    Responsabilidades:
    - Cargar las definiciones de tools desde disco al iniciar
    - Proveer las descripciones de tools al Harness (para el system prompt)
    - Resolver el nombre de una tool a su ToolDefinition (para el Bridge)
    """

    def __init__(self, tools_path: Path | str = "tools"):
        self.tools_path = Path(tools_path)
        self._tools: dict[str, ToolDefinition] = {}  # name → definición
        self.load()

    # ------------------------------------------------------------------
    # Carga desde disco
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carga el índice de tools. Soporta dos modos:
        1. Via _registry.yaml: lista explícita de tools habilitadas
        2. Discovery automático: escanea subdirectorios con meta.md
        """
        self._tools.clear()
        if not self.tools_path.exists():
            return

        registry_file = self.tools_path / "_registry.yaml"
        if registry_file.exists():
            self._load_from_registry(registry_file)
        else:
            self._discover_tools()

    def _load_from_registry(self, registry_file: Path) -> None:
        """Carga tools listadas en _registry.yaml."""
        try:
            with open(registry_file, encoding="utf-8") as f:
                registry = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[ToolRegistry] Error al leer _registry.yaml: {e}")
            return

        tools_list = registry.get("tools", [])
        for entry in tools_list:
            if isinstance(entry, str):
                tool_dir = self.tools_path / entry
            elif isinstance(entry, dict):
                tool_dir = self.tools_path / entry.get("path", entry.get("name", ""))
            else:
                continue

            tool = self._load_tool_dir(tool_dir)
            if tool:
                self._tools[tool.name] = tool

    def _discover_tools(self) -> None:
        """Escanea subdirectorios buscando meta.md automáticamente."""
        for tool_dir in self.tools_path.iterdir():
            if tool_dir.is_dir() and not tool_dir.name.startswith("_"):
                tool = self._load_tool_dir(tool_dir)
                if tool:
                    self._tools[tool.name] = tool

    def _load_tool_dir(self, tool_dir: Path) -> ToolDefinition | None:
        """Parsea el meta.md de un directorio de tool."""
        meta_path = tool_dir / "meta.md"
        if not meta_path.exists():
            return None

        try:
            post = frontmatter.load(str(meta_path))
            meta = post.metadata

            name = str(meta.get("name", tool_dir.name))
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
                executable = tool_dir / "run.py"
            elif runtime == "shell":
                executable = tool_dir / "run.sh"
            else:
                executable = tool_dir / "run.py"  # fallback

            if not executable.exists():
                print(f"[ToolRegistry] Advertencia: {executable} no existe para tool '{name}'")

            return ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=output_schema,
                executable_path=executable,
                runtime=runtime,
                tags=tags,
            )

        except Exception as e:
            print(f"[ToolRegistry] Error al cargar tool en {tool_dir}: {e}")
            return None

    def reload(self) -> None:
        """Recarga el catálogo desde disco (útil en desarrollo)."""
        self.load()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> ToolDefinition | None:
        """Devuelve la definición de una tool por nombre."""
        return self._tools.get(tool_name)

    def get_tool_descriptions(self) -> list[dict]:
        """
        Genera la lista de descriptores de tools para el system prompt del LLM.
        Formato compatible con OpenAI tool calling / Ollama.
        """
        return [tool.to_tool_dict() for tool in self._tools.values()]

    def format_for_prompt(self) -> str:
        """Genera texto legible de las tools disponibles para el system prompt."""
        if not self._tools:
            return ""
        lines = ["## Tools disponibles\n"]
        for tool in self._tools.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)

    def search_by_tags(self, query: str, max_results: int = 3) -> list[ToolDefinition]:
        """Búsqueda de tools por tags (para integración con el Splay Tree)."""
        normalized_query = self._normalize(query)
        query_words = set(re.findall(r"\b\w{2,}\b", normalized_query))
        if not query_words:
            return list(self._tools.values())[:max_results]

        scored: list[tuple[int, ToolDefinition]] = []
        for tool in self._tools.values():
            tag_set = set(self._normalize(t) for t in tool.tags)
            matches = len(query_words & tag_set)
            if matches > 0:
                scored.append((matches, tool))

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
    def tools(self) -> dict[str, ToolDefinition]:
        return self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolDefinition]:
        return iter(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools
