"""
iico_core/llm_client.py
=======================
Providers de LLM unificados detrás de un Protocol estándar.
Ambos (Ollama y OpenAI-compatible) implementan la misma interfaz,
lo que permite intercambiarlos sin modificar el Harness.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Protocol, runtime_checkable

import httpx

from .types import ChatMessage


# ---------------------------------------------------------------------------
# Protocol (interfaz pública del Harness)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """
    Contrato que deben implementar todos los providers de LLM.
    El system_prompt es un parámetro explícito (no hardcodeado),
    porque el Harness lo construye dinámicamente en cada turno.
    """

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Devuelve fragmentos de texto generados por el LLM (streaming)."""
        ...

    async def fetch_models(self) -> list[str]:
        """Lista los modelos disponibles en el endpoint."""
        ...


# ---------------------------------------------------------------------------
# Provider: Ollama
# ---------------------------------------------------------------------------

class OllamaClient:
    """Provider para Ollama (API /api/chat)."""

    def __init__(self, endpoint: str, model: str, temperature: float = 0.7):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._chat_url = f"{self.endpoint}/api/chat"
        self._tags_url = f"{self.endpoint}/api/tags"

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        # Construir el historial con el system prompt del Harness al inicio
        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages += [m.to_dict() for m in messages]

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        timeout = httpx.Timeout(120.0, connect=5.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream("POST", self._chat_url, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk:
                            try:
                                data = json.loads(chunk)
                                content = data.get("message", {}).get("content")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
            except httpx.ConnectError:
                yield f"\n[Sistema: No se pudo conectar a {self._chat_url}. ¿Está Ollama corriendo?]"
            except httpx.HTTPStatusError as e:
                yield f"\n[Sistema: Error HTTP {e.response.status_code} desde {self._chat_url}]"
            except Exception as e:
                yield f"\n[Sistema: Error inesperado con Ollama: {e}]"

    async def fetch_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._tags_url)
                if r.status_code == 200:
                    return [m.get("name") for m in r.json().get("models", [])]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Provider: OpenAI-compatible (llama.cpp, LM Studio, vLLM, etc.)
# ---------------------------------------------------------------------------

class OpenAIClient:
    """Provider compatible con la API de OpenAI (/v1/chat/completions)."""

    def __init__(self, endpoint: str, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self._chat_url = f"{base}/chat/completions"
        self._models_url = f"{base}/models"

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
    ) -> AsyncGenerator[str, None]:
        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages += [m.to_dict() for m in messages]

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": True,
            "temperature": self.temperature,
        }
        timeout = httpx.Timeout(120.0, connect=5.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream("POST", self._chat_url, json=payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk.startswith("data: "):
                            data_str = chunk[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices and "delta" in choices[0]:
                                    content = choices[0]["delta"].get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
            except httpx.ConnectError:
                yield f"\n[Sistema: No se pudo conectar a {self._chat_url}.]"
            except httpx.HTTPStatusError as e:
                yield f"\n[Sistema: Error HTTP {e.response.status_code} desde {self._chat_url}]"
            except Exception as e:
                yield f"\n[Sistema: Error inesperado: {e}]"

    async def fetch_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(self._models_url)
                if r.status_code == 200:
                    return [m.get("id") for m in r.json().get("data", [])]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_client(provider_type: str, endpoint: str, model: str, temperature: float = 0.7) -> LLMClient:
    """
    Crea el cliente correcto según el tipo de provider.
    Uso:
        client = create_client("ollama", "http://localhost:11434", "qwen2.5:7b")
        client = create_client("openai", "http://localhost:8080", "local-model")
    """
    if provider_type == "openai":
        return OpenAIClient(endpoint, model, temperature)
    return OllamaClient(endpoint, model, temperature)
