"""iico_core — Harness de gestión de contexto para agentes LLM en baja VRAM."""

from .harness import Harness
from .types import (
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    ProviderConfig,
    SkillDefinition,
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
    # Types
    "ChatMessage",
    "HarnessConfig",
    "HarnessEvent",
    "HarnessEventType",
    "ProviderConfig",
    "SkillDefinition",
    "ToolResult",
    # Fase 2
    "SkillRegistry",
    "ShellBridge",
    "SplayTree",
    "SplayCacheMetrics",
]
