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
    SkillDefinition,
    TaskGoal,
    TaskStatus,
    TaskTemplate,
    ToolResult,
)
from .llm_client import create_client
from .memory.active import SkillRegistry
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
    "SkillDefinition",
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
    "SkillRegistry",
    "ShellBridge",
    "SplayTree",
    "SplayCacheMetrics",
]
