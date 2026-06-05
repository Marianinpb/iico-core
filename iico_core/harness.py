"""
iico_core/harness.py
====================
Orquestador central del sistema iico-agent.

El Harness es el único punto de contacto entre el UI y todo el núcleo.
El UI solo llama a `process_input()` y consume los `HarnessEvent` que devuelve.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from .llm_client import LLMClient, create_client
from .memory.passive import PassiveMemory
from .types import (
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    ProviderConfig,
)


class Harness:
    """
    Orquestador principal del iico-agent.

    Responsabilidades en Fase 1:
    - Gestionar el historial de mensajes
    - Construir el system prompt dinámico con contexto filtrado
    - Llamar al LLM y emitir HarnessEvents
    - Manejar comandos slash (/)
    """

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.llm: LLMClient = create_client(
            provider_type=config.provider.type,
            endpoint=config.provider.endpoint,
            model=config.provider.model,
            temperature=config.provider.temperature,
        )
        self.passive_memory = PassiveMemory(config.memory_path)
        self.history: list[ChatMessage] = []

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    async def process_input(
        self,
        user_text: str,
    ) -> AsyncGenerator[HarnessEvent, None]:
        """
        Punto de entrada único para cualquier UI.
        Emite HarnessEvents que el UI consume para renderizar.
        """
        text = user_text.strip()
        if not text:
            return

        # Manejar comandos slash
        if text.startswith("/"):
            async for event in self._handle_command(text):
                yield event
            return

        # Mensaje normal → LLM
        self.history.append(ChatMessage(role="user", content=text))

        system_prompt = self.build_system_prompt(query=text)

        full_response = ""
        try:
            async for token in self.llm.chat_stream(self.history, system_prompt):
                full_response += token
                yield HarnessEvent(type=HarnessEventType.TOKEN, payload=token)
        except Exception as e:
            yield HarnessEvent(type=HarnessEventType.ERROR, payload=str(e))
            self.history.pop()   # Revertir el mensaje de usuario si falló
            return

        self.history.append(ChatMessage(role="assistant", content=full_response))
        yield HarnessEvent(type=HarnessEventType.DONE, payload=full_response)

    # ------------------------------------------------------------------
    # System prompt dinámico
    # ------------------------------------------------------------------

    def build_system_prompt(self, query: str = "") -> str:
        """
        Construye el system prompt inyectando solo el contexto relevante.
        
        Fase 1: tags YAML deterministas.
        Fase 2: se agregará búsqueda por embeddings ONNX.
        """
        parts = [self.config.system_prompt_base]

        if self.config.use_passive_memory and query:
            relevant = self.passive_memory.get_relevant(query, method="tags")
            if relevant:
                budgeted = self.passive_memory.apply_token_budget(
                    relevant,
                    max_tokens=self.config.token_budget // 2,  # Mitad del budget para contexto
                )
                context_text = self.passive_memory.format_for_prompt(budgeted)
                if context_text:
                    parts.append(context_text)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Comandos slash
    # ------------------------------------------------------------------

    async def _handle_command(
        self, text: str
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Procesa comandos slash como /model, /clear, /memory, etc."""
        parts = text.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/clear":
            self.history.clear()
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload="Historial de conversación borrado.",
            )

        elif cmd == "/memory":
            count = len(self.passive_memory)
            if count == 0:
                msg = "La memoria pasiva está vacía. Agrega archivos .md a memory_store/."
            else:
                ids = ", ".join(n.id for n in self.passive_memory)
                msg = f"Memoria pasiva: {count} nota(s) cargada(s).\nNotas: {ids}"
            yield HarnessEvent(type=HarnessEventType.SYSTEM, payload=msg)

        elif cmd == "/memory-reload":
            self.passive_memory.reload()
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload=f"Memoria pasiva recargada: {len(self.passive_memory)} nota(s).",
            )

        else:
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload=f"Comando desconocido '{cmd}'. Comandos: /clear, /memory, /memory-reload",
            )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def change_provider(self, provider: ProviderConfig) -> None:
        """Cambia el proveedor de LLM en caliente."""
        self.config.provider = provider
        self.llm = create_client(
            provider_type=provider.type,
            endpoint=provider.endpoint,
            model=provider.model,
            temperature=provider.temperature,
        )

    async def fetch_models(self) -> list[str]:
        """Lista los modelos disponibles en el endpoint activo."""
        return await self.llm.fetch_models()

    def clear_history(self) -> None:
        self.history.clear()

    @property
    def model_name(self) -> str:
        return self.config.provider.model
