"""
tests/test_shell_bridge.py
===========================
Tests del ShellBridge: ejecución de tools como subprocesos.
"""

import json
import sys
from pathlib import Path

import pytest

from iico_core.bridge.shell import ShellBridge
from iico_core.types import ToolDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bridge():
    return ShellBridge(default_timeout=10.0)


def make_python_tool(tmp_path, script_content: str, name: str = "test_tool") -> ToolDefinition:
    """Crea una ToolDefinition temporal con un script Python dado."""
    script = tmp_path / "run.py"
    script.write_text(script_content, encoding="utf-8")
    return ToolDefinition(
        name=name,
        description="Test tool",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object"},
        executable_path=script,
        runtime="python",
    )


# ---------------------------------------------------------------------------
# Ejecución exitosa
# ---------------------------------------------------------------------------

def test_execute_simple_python(bridge, tmp_path):
    """Script que lee args de stdin y escribe resultado en stdout."""
    tool = make_python_tool(
        tmp_path,
        'import json, sys\n'
        'args = json.loads(sys.stdin.read())\n'
        'print(json.dumps({"echo": args.get("value", "")}))\n'
    )
    result = bridge.execute(tool, {"value": "hello"})
    assert result.success
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["echo"] == "hello"


def test_execute_duration_tracked(bridge, tmp_path):
    """La duración debe ser mayor que 0."""
    tool = make_python_tool(tmp_path, 'print("ok")\n')
    result = bridge.execute(tool, {})
    assert result.duration_ms >= 0.0


def test_execute_stdout_captured(bridge, tmp_path):
    tool = make_python_tool(tmp_path, 'print("resultado_capturado")\n')
    result = bridge.execute(tool, {})
    assert "resultado_capturado" in result.output


def test_execute_stderr_captured(bridge, tmp_path):
    tool = make_python_tool(
        tmp_path,
        'import sys\nprint("error_message", file=sys.stderr)\nsys.exit(1)\n'
    )
    result = bridge.execute(tool, {})
    assert result.exit_code == 1
    assert "error_message" in result.error
    assert not result.success


# ---------------------------------------------------------------------------
# Errores y casos límite
# ---------------------------------------------------------------------------

def test_execute_missing_script(bridge, tmp_path):
    """Script que no existe → exit code 1 con mensaje de error."""
    tool = ToolDefinition(
        name="missing",
        description="",
        input_schema={},
        output_schema={},
        executable_path=tmp_path / "nonexistent.py",
        runtime="python",
    )
    result = bridge.execute(tool, {})
    assert not result.success
    assert result.exit_code == 1
    assert "no encontrado" in result.error.lower() or "not found" in result.error.lower()


def test_execute_timeout(tmp_path):
    """Script que duerme → debe triggear timeout."""
    bridge = ShellBridge(default_timeout=0.5)
    tool = make_python_tool(tmp_path, 'import time\ntime.sleep(10)\n')
    result = bridge.execute(tool, {})
    assert result.exit_code == 124  # Convención de timeout
    assert not result.success
    assert "timeout" in result.error.lower() or "excedió" in result.error.lower()


def test_execute_unknown_runtime(bridge, tmp_path):
    """Runtime desconocido → error descriptivo."""
    tool = ToolDefinition(
        name="weird",
        description="",
        input_schema={},
        output_schema={},
        executable_path=tmp_path / "run.js",
        runtime="javascript",
    )
    result = bridge.execute(tool, {})
    assert not result.success
    assert "runtime" in result.error.lower()


def test_execute_calculator_tool_integration(bridge, tmp_path):
    """Prueba el protocolo JSON stdin/stdout con la tool calculator real."""
    # Script idéntico al de la tool calculator
    calc_script = (
        'import json, math, sys\n'
        'args = json.loads(sys.stdin.read())\n'
        'expr = args.get("expression", "")\n'
        'safe_globals = {"__builtins__": {}, **{n: getattr(math, n) for n in dir(math) if not n.startswith("_")}}\n'
        'result = float(eval(expr, safe_globals, {}))\n'
        'print(json.dumps({"result": result, "expression": expr}))\n'
    )
    tool = make_python_tool(tmp_path, calc_script, name="calculator")
    result = bridge.execute(tool, {"expression": "2 + 2 * 3"})
    assert result.success
    data = json.loads(result.output)
    assert data["result"] == 8.0


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------

def test_tool_result_success_property(bridge, tmp_path):
    tool = make_python_tool(tmp_path, 'import sys\nsys.exit(0)\n')
    result = bridge.execute(tool, {})
    assert result.success is True


def test_tool_result_to_dict(bridge, tmp_path):
    tool = make_python_tool(tmp_path, 'print("x")\n')
    result = bridge.execute(tool, {})
    d = result.to_dict()
    assert "tool" in d
    assert "output" in d
    assert "exit_code" in d
    assert "success" in d
    assert "duration_ms" in d
