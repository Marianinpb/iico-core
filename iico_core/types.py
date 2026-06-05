"""
iico_core/types.py
==================
Dataclasses y tipos compartidos que definen el contrato entre el núcleo (Harness)
y cualquier interfaz de usuario (TUI, Open WebUI, etc.).
"""

from __future__ import annotations

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
    role: str          # "user", "assistant" o "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, str]:
        """Formato compatible con la API de Ollama / OpenAI."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuración de un proveedor de LLM."""
    type: str          # "ollama" | "openai"
    endpoint: str
    model: str
    temperature: float = 0.7


@dataclass
class HarnessConfig:
    """Configuración completa del Harness."""
    # Proveedor LLM activo
    provider: ProviderConfig

    # Rutas de datos
    memory_path: Path = field(default_factory=lambda: Path("memory_store"))
    skills_path: Path = field(default_factory=lambda: Path("skills"))

    # Comportamiento del Arnés
    token_budget: int = 4096          # Máximo de tokens a inyectar en el system prompt
    mode: str = "harness"             # "harness" | "baseline"

    # Flags de características (para experimentos A/B)
    use_passive_memory: bool = True
    use_splay_tree: bool = True
    use_embedding_search: bool = False   # Requiere ONNX, desactivado en Fase 1
    use_react_loop: bool = False         # Se activa en Fase 3

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
    TOKEN         = "token"           # Fragmento de texto generado por el LLM
    DONE          = "done"            # Respuesta completa
    ERROR         = "error"           # Error recuperable
    SYSTEM        = "system"          # Mensaje de sistema (ej: "Modelo cambiado a X")
    THINKING      = "thinking"        # El agente está razonando (Fase 3)
    SKILL_START   = "skill_start"     # Inicio de ejecución de una skill
    SKILL_DONE    = "skill_done"      # Fin de ejecución de una skill
    PLAN_UPDATE   = "plan_update"     # Actualización del plan de tareas


@dataclass
class HarnessEvent:
    """Evento emitido por el Harness hacia el UI."""
    type: HarnessEventType
    payload: Any = None               # str para TOKEN/ERROR/SYSTEM, dict para PLAN_UPDATE, etc.


# ---------------------------------------------------------------------------
# Tool Calls (Fase 2+)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """Representación de una invocación de skill por parte del LLM."""
    name: str
    args: dict[str, Any]
    result: Any = None
    success: bool = True
    error: str | None = None
