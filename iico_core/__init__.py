"""iico_core — Harness de gestión de contexto para agentes LLM en baja VRAM."""

from .harness import Harness
from .types import (
    AgentState,
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    LLMResponse,
    LLMToolCall,
    ProviderConfig,
    SDDDocument,
    ToolDefinition,
    TaskGoal,
    TaskStatus,
    TaskTemplate,
    ToolResult,
)
from .llm_client import create_client
from .memory.active import ToolRegistry
from .memory.skills import SkillLibrary
from .bridge.shell import ShellBridge
from .index.splay_tree import SplayTree, SplayCacheMetrics

__all__ = [
    # Core
    "Harness",
    "create_client",
    # Types Fase 1-2
    "ChatMessage",
    "HarnessConfig",
    "HarnessEvent",
    "HarnessEventType",
    "ProviderConfig",
    "ToolDefinition",
    "ToolResult",
    # Types Fase 3
    "AgentState",
    "LLMResponse",
    "LLMToolCall",
    "SDDDocument",
    "TaskGoal",
    "TaskStatus",
    "TaskTemplate",
    # Subsistemas
    "SkillLibrary",
    "ToolRegistry",
    "ShellBridge",
    "SplayTree",
    "SplayCacheMetrics",
]
