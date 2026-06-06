"""
tests/test_skill_registry.py
=============================
Tests del SkillRegistry y SkillDefinition.
"""

import json
from pathlib import Path

import pytest

from iico_core.memory.active import SkillRegistry
from iico_core.types import SkillDefinition


# ---------------------------------------------------------------------------
# Fixtures: directorio temporal de skills
# ---------------------------------------------------------------------------

@pytest.fixture
def skills_dir(tmp_path):
    """Crea una estructura de skills temporal para los tests."""
    # Skill 1: calculator
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

    # Skill 2: greeter
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
        "skills:\n  - calculator\n  - greeter\n",
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Carga desde registry
# ---------------------------------------------------------------------------

def test_load_from_registry(skills_dir):
    registry = SkillRegistry(skills_dir)
    assert len(registry) == 2


def test_skill_names_loaded(skills_dir):
    registry = SkillRegistry(skills_dir)
    assert "calculator" in registry
    assert "greeter" in registry


def test_skill_definition_fields(skills_dir):
    registry = SkillRegistry(skills_dir)
    calc = registry.get("calculator")
    assert calc is not None
    assert calc.name == "calculator"
    assert "matematicas" in calc.tags
    assert calc.runtime == "python"
    assert calc.executable_path.name == "run.py"


def test_get_missing_skill(skills_dir):
    registry = SkillRegistry(skills_dir)
    result = registry.get("nonexistent_skill")
    assert result is None


# ---------------------------------------------------------------------------
# Discovery automático (sin _registry.yaml)
# ---------------------------------------------------------------------------

def test_autodiscovery(tmp_path):
    """Si no hay _registry.yaml, descubre skills automáticamente."""
    skill_dir = tmp_path / "my_skill"
    skill_dir.mkdir()
    (skill_dir / "meta.md").write_text(
        "---\nname: my_skill\ndescription: Test.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    (skill_dir / "run.py").write_text("pass\n")

    registry = SkillRegistry(tmp_path)
    assert "my_skill" in registry


def test_autodiscovery_ignores_underscore_dirs(tmp_path):
    """Directorios que empiezan con _ se ignoran en discovery."""
    hidden_dir = tmp_path / "_internal"
    hidden_dir.mkdir()
    (hidden_dir / "meta.md").write_text(
        "---\nname: internal\ndescription: Hidden.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    registry = SkillRegistry(tmp_path)
    assert "internal" not in registry


# ---------------------------------------------------------------------------
# get_tool_descriptions
# ---------------------------------------------------------------------------

def test_get_tool_descriptions_format(skills_dir):
    registry = SkillRegistry(skills_dir)
    tools = registry.get_tool_descriptions()
    assert isinstance(tools, list)
    assert len(tools) == 2
    for tool in tools:
        assert tool["type"] == "function"
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]


def test_to_tool_dict_format(skills_dir):
    registry = SkillRegistry(skills_dir)
    calc = registry.get("calculator")
    d = calc.to_tool_dict()
    assert d["function"]["name"] == "calculator"
    assert "expression" in d["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# format_for_prompt
# ---------------------------------------------------------------------------

def test_format_for_prompt_nonempty(skills_dir):
    registry = SkillRegistry(skills_dir)
    text = registry.format_for_prompt()
    assert "calculator" in text
    assert "greeter" in text


def test_format_for_prompt_empty_registry():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        registry = SkillRegistry(d)
        text = registry.format_for_prompt()
        assert text == ""


# ---------------------------------------------------------------------------
# Búsqueda por tags
# ---------------------------------------------------------------------------

def test_search_by_tags_finds_match(skills_dir):
    registry = SkillRegistry(skills_dir)
    results = registry.search_by_tags("necesito hacer un calculo matematicas")
    names = [s.name for s in results]
    assert "calculator" in names


def test_search_by_tags_no_match_returns_all(skills_dir):
    """Si no hay palabras clave útiles, retorna las primeras skills."""
    registry = SkillRegistry(skills_dir)
    results = registry.search_by_tags("xy zw", max_results=5)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

def test_reload(skills_dir):
    registry = SkillRegistry(skills_dir)
    initial = len(registry)
    # Agregar skill nueva
    new_dir = skills_dir / "new_skill"
    new_dir.mkdir()
    (new_dir / "meta.md").write_text(
        "---\nname: new_skill\ndescription: Nueva.\nruntime: python\n---\n",
        encoding="utf-8",
    )
    # Actualizar registry.yaml
    (skills_dir / "_registry.yaml").write_text(
        "skills:\n  - calculator\n  - greeter\n  - new_skill\n",
        encoding="utf-8",
    )
    registry.reload()
    assert len(registry) == initial + 1


# ---------------------------------------------------------------------------
# Iteración
# ---------------------------------------------------------------------------

def test_iteration(skills_dir):
    registry = SkillRegistry(skills_dir)
    names = [s.name for s in registry]
    assert set(names) == {"calculator", "greeter"}
