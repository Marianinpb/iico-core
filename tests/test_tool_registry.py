"""
tests/test_tool_registry.py
=============================
Tests del ToolRegistry y ToolDefinition.
"""

import json
from pathlib import Path

import pytest

from iico_core.memory.active import ToolRegistry
from iico_core.types import ToolDefinition


# ---------------------------------------------------------------------------
# Fixtures: directorio temporal de tools
# ---------------------------------------------------------------------------

@pytest.fixture
def tools_dir(tmp_path):
    """Crea una estructura de tools temporal para los tests."""
    # Tool 1: calculator
    calc_dir = tmp_path / "calculator"
    calc_dir.mkdir()
    (calc_dir / "meta.md").write_text(
        "---\n"
        "name: calculator\n"
        "description: Evalúa expresiones matemáticas.\n"
        "runtime: python\n"
        "tags: [matematicas, calculo]\n"
        "input_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    expression:\n"
        "      type: string\n"
        "  required: [expression]\n"
        "output_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    result:\n"
        "      type: number\n"
        "---\n"
        "# Calculator\nEvalúa expresiones matemáticas.\n",
        encoding="utf-8",
    )
    (calc_dir / "run.py").write_text("import sys\nprint('ok')\n", encoding="utf-8")

    # Tool 2: greeter
    greet_dir = tmp_path / "greeter"
    greet_dir.mkdir()
    (greet_dir / "meta.md").write_text(
        "---\n"
        "name: greeter\n"
        "description: Saluda al usuario.\n"
        "runtime: python\n"
        "tags: [saludo, utilidad]\n"
        "input_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    name:\n"
        "      type: string\n"
        "  required: [name]\n"
        "output_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    message:\n"
        "      type: string\n"
        "---\n",
        encoding="utf-8",
    )
    (greet_dir / "run.py").write_text("import sys\nprint('hello')\n", encoding="utf-8")

    # Registry
    (tmp_path / "_registry.yaml").write_text(
        "tools:\n  - calculator\n  - greeter\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Carga desde registry
# ---------------------------------------------------------------------------

def test_load_from_registry(tools_dir):
    registry = ToolRegistry(tools_dir)
    assert len(registry) == 2


def test_tool_names_loaded(tools_dir):
    registry = ToolRegistry(tools_dir)
    assert "calculator" in registry
    assert "greeter" in registry


def test_tool_definition_fields(tools_dir):
    registry = ToolRegistry(tools_dir)
    calc = registry.get("calculator")
    assert calc is not None
    assert calc.name == "calculator"
    assert "matematicas" in calc.tags
    assert calc.runtime == "python"
    assert calc.executable_path.name == "run.py"


def test_get_missing_tool(tools_dir):
    registry = ToolRegistry(tools_dir)
    result = registry.get("nonexistent_tool")
    assert result is None


# ---------------------------------------------------------------------------
# Discovery automático (sin _registry.yaml)
# ---------------------------------------------------------------------------

def test_autodiscovery(tmp_path):
    """Si no hay _registry.yaml, descubre tools automáticamente."""
    tool_dir = tmp_path / "my_tool"
    tool_dir.mkdir()
    (tool_dir / "meta.md").write_text(
        "---\nname: my_tool\ndescription: Test.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    (tool_dir / "run.py").write_text("pass\n")

    registry = ToolRegistry(tmp_path)
    assert "my_tool" in registry


def test_autodiscovery_ignores_underscore_dirs(tmp_path):
    """Directorios que empiezan con _ se ignoran en discovery."""
    hidden_dir = tmp_path / "_internal"
    hidden_dir.mkdir()
    (hidden_dir / "meta.md").write_text(
        "---\nname: internal\ndescription: Hidden.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(tmp_path)
    assert "internal" not in registry


# ---------------------------------------------------------------------------
# get_tool_descriptions
# ---------------------------------------------------------------------------

def test_get_tool_descriptions_format(tools_dir):
    registry = ToolRegistry(tools_dir)
    tools = registry.get_tool_descriptions()
    assert isinstance(tools, list)
    assert len(tools) == 2
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_to_tool_dict_format(tools_dir):
    registry = ToolRegistry(tools_dir)
    calc = registry.get("calculator")
    d = calc.to_tool_dict()
    assert d["function"]["name"] == "calculator"
    assert "expression" in d["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt_nonempty(tools_dir):
    registry = ToolRegistry(tools_dir)
    text = registry.format_for_prompt()
    assert "calculator" in text
    assert "greeter" in text


def test_format_for_prompt_empty_registry():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        registry = ToolRegistry(d)
        text = registry.format_for_prompt()
        assert text == ""


# ---------------------------------------------------------------------------
# Búsqueda por tags
# ---------------------------------------------------------------------------

def test_search_by_tags_finds_match(tools_dir):
    registry = ToolRegistry(tools_dir)
    results = registry.search_by_tags("necesito hacer un calculo matematicas")
    names = [t.name for t in results]
    assert "calculator" in names


def test_search_by_tags_no_match_returns_all(tools_dir):
    """Si no hay palabras clave útiles, retorna las primeras tools."""
    registry = ToolRegistry(tools_dir)
    results = registry.search_by_tags("xy zw", max_results=5)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def test_reload(tools_dir):
    registry = ToolRegistry(tools_dir)
    initial = len(registry)
    # Agregar tool nueva
    new_dir = tools_dir / "new_tool"
    new_dir.mkdir()
    (new_dir / "meta.md").write_text(
        "---\nname: new_tool\ndescription: Nueva.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    # Actualizar registry.yaml
    (tools_dir / "_registry.yaml").write_text(
        "tools:\n  - calculator\n  - greeter\n  - new_tool\n",
        encoding="utf-8",
    )
    registry.reload()
    assert len(registry) == initial + 1


# ---------------------------------------------------------------------------
# Iteración
# ---------------------------------------------------------------------------

def test_iteration(tools_dir):
    registry = ToolRegistry(tools_dir)
    names = [t.name for t in registry]
    assert set(names) == {"calculator", "greeter"}
