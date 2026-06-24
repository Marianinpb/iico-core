"""
iico_core/types.py
==================
Dataclasses y tipos compartidos que definen el contrato entre el núcleo (Harness)
y cualquier interfaz de usuario (TUI, Open WebUI, etc.).

Fase 2: se agregan ToolDefinition y ToolResult para el sistema de tools.
Fase 3: se agregan AgentState, TaskTemplate, SDDDocument y LLMResponse para
el flujo SDD y el bucle ReAct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Mensajes
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """Representa un mensaje en el historial de conversación."""
    role: str          # "user", "assistant", "system" o "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[LLMToolCall] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Formato compatible con la API de Ollama / OpenAI."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)}
                } for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuración de un proveedor de LLM."""
    type: str          # "ollama" | "openai" | "deepseek"
    endpoint: str
    model: str
    temperature: float = 0.7
    api_key: str = ""  # Clave API para proveedores que requieren auth (OpenAI, DeepSeek)


@dataclass
class HarnessConfig:
    """Configuración completa del Harness."""
    # Proveedor LLM activo
    provider: ProviderConfig

    # Rutas de datos
    memory_path: Path = field(default_factory=lambda: Path("memory_store"))
    tools_path: Path = field(default_factory=lambda: Path("tools"))
    skills_path: Path = field(default_factory=lambda: Path("skills"))

    # Comportamiento del Arnés
    token_budget: int = 4096          # Máximo de tokens a inyectar en el system prompt
    mode: str = "harness"             # "harness" | "baseline"

    # Flags de características (para experimentos A/B)
    use_passive_memory: bool = True
    use_splay_tree: bool = True
    use_embedding_search: bool = False   # Requiere iico-core[embeddings] instalado
    use_react_loop: bool = False         # Se activa en Fase 3
    use_tools: bool = False             # Activa ToolRegistry + ShellBridge
    use_skills: bool = False            # Activa SkillLibrary (workflows .md)

    # --- Chunking (Característica 4: Memoria Particionada) ---
    use_chunking: bool = False  # Activar chunking en vez de notas completas

    # --- Splay Tree (Nivel 2) ---
    splay_cache_size: int = 50           # Máximo de nodos en el Splay Tree
    splay_peek_top: int = 5              # Nodos a consultar sin splayear (hit check rápido)

    # --- EmbeddingIndex (Nivel 1) ---
    embedding_threshold: float = 0.50   # Umbral mínimo de similitud del coseno (MiniLM en español: 0.4-0.65)
    max_context_notes: int = 5          # Máximo de notas a inyectar en el prompt

    # --- Tools ---
    tool_timeout: float = 30.0         # Timeout en segundos para ejecución de tools
    require_command_confirmation: bool = True   # Pedir confirmación antes de run_command

    # System prompt base
    system_prompt_base: str = (
        "Eres iico-agent. Este es tu nombre y tu identidad. No eres ningún modelo de lenguaje genérico ni reveles "
        "el nombre del modelo subyacente que te ejecuta. Eres un asistente técnico avanzado desarrollado como parte "
        "de un proyecto de investigación de Maestría en Ingeniería. Tu propósito es ayudar con tareas de ingeniería, "
        "procesamiento de datos y análisis técnico. Cuando alguien te pregunte quién eres, cómo te llamas, o de dónde "
        "vienes, responde siempre como iico-agent usando el contexto que tienes en memoria. "
        "Responde de forma precisa y concisa, siempre en el idioma del usuario."
    )


# ---------------------------------------------------------------------------
# Eventos emitidos por el Harness → consumidos por el UI
# ---------------------------------------------------------------------------

class HarnessEventType(Enum):
    TOKEN              = "token"           # Fragmento de texto generado por el LLM
    DONE               = "done"            # Respuesta completa
    ERROR              = "error"           # Error recuperable
    SYSTEM             = "system"          # Mensaje de sistema
    THINKING           = "thinking"        # El agente está razonando
    TOOL_START        = "tool_start"     # Inicio de ejecución de una tool
    TOOL_DONE         = "tool_done"      # Fin de ejecución de una tool
    PLAN_UPDATE        = "plan_update"     # Actualización del plan de tareas
    # --- Fase 3: SDD y ReAct ---
    SDD_STARTED        = "sdd_started"     # Se inició un flujo SDD
    SDD_QUESTION       = "sdd_question"    # El agente pregunta algo al usuario
    PLAN_PROPOSED      = "plan_proposed"   # Plan listo para aprobación
    TASK_STARTED       = "task_started"    # Inicio de ejecución de una task
    TASK_COMPLETED     = "task_completed"  # Task completada
    TASK_FAILED        = "task_failed"     # Task falló
    GOAL_VERIFIED      = "goal_verified"   # Meta comprobada
    STATE_CHANGED      = "state_changed"   # Cambio de AgentState
    TOKEN_USAGE        = "token_usage"     # Uso de tokens actualizados
    COMMAND_APPROVAL_REQUIRED = "command_approval_required"  # Pedir al usuario si puede ejecutar un comando


@dataclass
class HarnessEvent:
    """Evento emitido por el Harness hacia el UI."""
    type: HarnessEventType
    payload: Any = None               # str para TOKEN/ERROR/SYSTEM, dict para PLAN_UPDATE, etc.


# ---------------------------------------------------------------------------
# Fase 3: Estado del Agente (Máquina de Estados)
# ---------------------------------------------------------------------------

class AgentState(Enum):
    """Estado actual del agente en el flujo SDD."""
    IDLE              = "idle"              # Esperando input del usuario
    INTERVIEWING      = "interviewing"      # Recopilando requisitos para el SDD
    PLANNING          = "planning"          # Generando plan de acción
    AWAITING_APPROVAL = "awaiting_approval" # Plan presentado, esperando aprobación
    EXECUTING         = "executing"         # Ejecutando tasks vía ReAct
    VERIFYING         = "verifying"         # Verificando metas de una task


class TaskStatus(Enum):
    """Estado de una tarea dentro del plan."""
    PENDING     = "pending"
    BLOCKED     = "blocked"      # Dependencias no satisfechas
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"


@dataclass
class TaskGoal:
    """Meta comprobable de una tarea."""
    description: str
    verification_tool: str | None = None  # Nombre de la tool para verificar
    verification_args: dict = field(default_factory=dict)
    met: bool = False


@dataclass
class TaskTemplate:
    """Plantilla de tarea generada por el TaskManager."""
    id: str                              # ej: "task_1"
    description: str
    goals: list[TaskGoal] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # IDs de tareas prerequisito
    status: TaskStatus = TaskStatus.PENDING
    tags: list[str] = field(default_factory=list)  # Para búsqueda por tags
    result_summary: str = ""             # Resumen del resultado al completar

    def is_ready(self, completed_ids: set[str]) -> bool:
        """¿Están todas las dependencias satisfechas?"""
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class SDDDocument:
    """Documento de Especificación (Spec-Driven Development)."""
    title: str
    description: str
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    raw_markdown: str = ""              # El SDD completo como nota consultable
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Fase 4: Chunking de Notas (Memoria Particionada)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Fragmento de una nota, generado por el Chunker.

    Cada nota de la memoria pasiva puede dividirse en múltiples chunks
    (uno por sección o párrafo semántico). Los chunks heredan prioridad
    y tags del parent, pero tienen contenido acotado (~50-300 tokens).
    """
    id: str                           # ej: "arquitectura_harness::splay-rotations"
    parent_note_id: str               # ej: "arquitectura_harness"
    title: str                        # ej: "Rotaciones Splay"
    content: str                      # texto del chunk (~50-300 tokens)
    tags: list[str] = field(default_factory=list)
    priority: int = 5                 # heredado del parent
    order: int = 0                    # posición secuencial dentro de la nota original
    source_path: "Path | None" = None       # ruta al .md del chunk
    embedding_path: "Path | None" = None    # ruta al .npy del chunk
    content_hash: str = ""            # sha256 del content

    def token_estimate(self) -> int:
        """Estimación rápida de tokens (1 token ≈ 4 caracteres en español)."""
        return len(self.content) // 4


# ---------------------------------------------------------------------------
# Fase 3: Respuesta del LLM con Tool Calls
# ---------------------------------------------------------------------------

@dataclass
class LLMToolCall:
    """Tool call emitido por el LLM (formato nativo Ollama/OpenAI)."""
    call_id: str                         # ID único de la llamada
    name: str                            # Nombre de la tool
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Respuesta completa del LLM (no streaming, para el ReAct loop)."""
    content: str                         # Texto de la respuesta
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str = "stop"          # "stop" | "tool_calls"
    usage: dict[str, int] = field(default_factory=dict) # ej: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# ---------------------------------------------------------------------------
# Tool Calls (Fase 2+)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Representación de una invocación de tool por parte del LLM."""
    name: str
    args: dict[str, Any]
    result: Any = None
    success: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Tools (Fase 2)
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """
    Definición de una tool cargada desde disco.
    Cada tool vive en tools/<nombre>/ con un meta.md y un run.py.
    """
    name: str                          # Identificador único, ej: "calculator"
    description: str                   # Lo que el LLM ve en el system prompt
    input_schema: dict[str, Any]       # JSON Schema de los argumentos de entrada
    output_schema: dict[str, Any]      # JSON Schema de la salida
    executable_path: Path              # Ruta al script/binario a ejecutar
    runtime: str = "python"           # "python" | "shell" | "native"
    tags: list[str] = field(default_factory=list)  # Tags para búsqueda semántica

    def to_tool_dict(self) -> dict[str, Any]:
        """Genera el descriptor de herramienta en formato compatible con LLMs (OpenAI/Ollama)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolResult:
    """
    Resultado de la ejecución de una tool por parte del ShellBridge.
    """
    tool_name: str
    output: str                        # stdout del proceso
    exit_code: int = 0
    error: str = ""                   # stderr del proceso
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool_name,
            "output": self.output,
            "exit_code": self.exit_code,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "success": self.success,
        }
