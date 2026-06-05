"""iico_core — Harness de gestión de contexto para agentes LLM en baja VRAM."""

from .harness import Harness
from .types import (
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    ProviderConfig,
)
from .llm_client import create_client

__all__ = [
    "Harness",
    "ChatMessage",
    "HarnessConfig",
    "HarnessEvent",
    "HarnessEventType",
    "ProviderConfig",
    "create_client",
]
