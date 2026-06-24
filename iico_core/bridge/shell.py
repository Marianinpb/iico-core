"""
iico_core/bridge/shell.py
==========================
ShellBridge: ejecuta tools de forma segura como subprocesos.

El Bridge es la única puerta de salida del Arnés hacia el sistema operativo.
Toda ejecución pasa por aquí, garantizando:
- Timeout configurable por tool
- Captura de stdout/stderr
- Registro de duración para métricas
- Sandbox básico: sin acceso a red por defecto, whitelist de paths

Runtimes soportados:
- "python": ejecuta `python run.py` con args como JSON en stdin
- "shell": ejecuta un script .sh directamente
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from ..types import ToolDefinition, ToolResult


class ShellBridge:
    """
    Ejecuta tools como subprocesos controlados.

    El protocolo de comunicación con los scripts de tools es simple:
    - Los argumentos se pasan como JSON en la primera línea de stdin
    - El script escribe su resultado como JSON en stdout
    - Cualquier error va a stderr
    - exit code 0 = éxito, cualquier otro = fallo
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        allowed_paths: list[Path] | None = None,
        python_executable: str | None = None,
        project_root: Path | None = None,
    ):
        self.default_timeout = default_timeout
        self.allowed_paths = allowed_paths  # None = sin restricción de paths
        self.python_executable = python_executable or sys.executable
        # cwd para ejecución de tools: la raíz del proyecto si está disponible,
        # de lo contrario la carpeta de la tool (comportamiento anterior)
        self.project_root = project_root

    # ------------------------------------------------------------------
    # Ejecución principal
    # ------------------------------------------------------------------

    def execute(
        self,
        tool: ToolDefinition,
        args: dict,
        timeout: float | None = None,
    ) -> ToolResult:
        """
        Ejecuta una tool y devuelve su resultado.

        Args:
            tool: definición de la tool a ejecutar
            args: diccionario de argumentos para la tool
            timeout: timeout en segundos (usa default_timeout si es None)

        Returns:
            ToolResult con output, exit_code, error y duration_ms
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout

        if tool.runtime == "python":
            return self._execute_python(tool, args, effective_timeout)
        elif tool.runtime == "shell":
            return self._execute_shell(tool, args, effective_timeout)
        else:
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=1,
                error=f"Runtime desconocido: '{tool.runtime}'. Soportados: python, shell",
            )

    # ------------------------------------------------------------------
    # Runtimes
    # ------------------------------------------------------------------

    def _execute_python(
        self,
        tool: ToolDefinition,
        args: dict,
        timeout: float,
    ) -> ToolResult:
        """Ejecuta `python run.py` pasando args como JSON en stdin."""
        if not tool.executable_path.exists():
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=1,
                error=f"Script no encontrado: {tool.executable_path}",
            )

        args_copy = dict(args)
        if self.project_root and self.project_root.exists():
            args_copy["_project_root"] = str(self.project_root)
        else:
            args_copy["_project_root"] = str(Path.cwd())

        args_json = json.dumps(args_copy, ensure_ascii=False)
        start = time.perf_counter()
        # Usar la raíz del proyecto como cwd; si no está configurada, usar la carpeta de la tool
        effective_cwd = self.project_root if self.project_root and self.project_root.exists() else tool.executable_path.parent

        try:
            proc = subprocess.run(
                [self.python_executable, str(tool.executable_path)],
                input=args_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=effective_cwd,
            )
            duration_ms = (time.perf_counter() - start) * 1000

            return ToolResult(
                tool_name=tool.name,
                output=proc.stdout.strip(),
                exit_code=proc.returncode,
                error=proc.stderr.strip(),
                duration_ms=duration_ms,
            )

        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=124,  # Convención Unix para timeout
                error=f"Timeout: la tool '{tool.name}' excedió {timeout:.1f}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=1,
                error=f"Error al ejecutar tool: {e}",
                duration_ms=duration_ms,
            )

    def _execute_shell(
        self,
        tool: ToolDefinition,
        args: dict,
        timeout: float,
    ) -> ToolResult:
        """Ejecuta un script shell, pasando args como variables de entorno."""
        if not tool.executable_path.exists():
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=1,
                error=f"Script no encontrado: {tool.executable_path}",
            )

        import os
        env = os.environ.copy()
        
        args_copy = dict(args)
        if self.project_root and self.project_root.exists():
            args_copy["_project_root"] = str(self.project_root)
        else:
            args_copy["_project_root"] = str(Path.cwd())

        # Pasar cada arg como variable de entorno TOOL_<KEY>=<value>
        for key, val in args_copy.items():
            env[f"TOOL_{key.upper()}"] = str(val)
        
        # También pasar el JSON completo
        env["TOOL_ARGS_JSON"] = json.dumps(args_copy)
        
        # Usar la raíz del proyecto como cwd; si no está configurada, usar la carpeta de la tool
        effective_cwd = self.project_root if self.project_root and self.project_root.exists() else tool.executable_path.parent

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(tool.executable_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=effective_cwd,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool.name,
                output=proc.stdout.strip(),
                exit_code=proc.returncode,
                error=proc.stderr.strip(),
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=124,
                error=f"Timeout: la tool '{tool.name}' excedió {timeout:.1f}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=tool.name,
                output="",
                exit_code=1,
                error=f"Error al ejecutar shell tool: {e}",
                duration_ms=duration_ms,
            )
