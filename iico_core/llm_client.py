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

from .types import ChatMessage, LLMResponse, LLMToolCall


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

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Envía mensajes con herramientas disponibles y devuelve la respuesta completa."""
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

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Llama a Ollama con tool calling nativo. Si el modelo no lo soporta, usa fallback por prompt."""
        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages += [m.to_dict() for m in messages]

        payload: dict = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        timeout = httpx.Timeout(120.0, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(self._chat_url, json=payload)

                # Si el modelo no soporta tools nativas → fallback por prompt
                if r.status_code == 400 and tools:
                    error_body = r.text
                    if "does not support tools" in error_body:
                        return await self._chat_with_tools_fallback(
                            payload_messages, system_prompt, tools, timeout
                        )

                r.raise_for_status()
                data = r.json()

            message = data.get("message", {})
            content = message.get("content") or ""
            finish_reason = "stop"
            tool_calls: list[LLMToolCall] = []

            raw_tools = message.get("tool_calls", [])
            if raw_tools:
                finish_reason = "tool_calls"
                for i, tc in enumerate(raw_tools):
                    fn = tc.get("function", {})
                    tool_calls.append(LLMToolCall(
                        call_id=str(i),
                        name=fn.get("name", ""),
                        args=fn.get("arguments") or {},
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            )
        except Exception as e:
            return LLMResponse(
                content=f"[Error en tool calling: {e}]",
                finish_reason="error",
            )

    async def _chat_with_tools_fallback(
        self,
        original_messages: list[dict],
        system_prompt: str,
        tools: list[dict],
        timeout: httpx.Timeout,
    ) -> LLMResponse:
        """
        Fallback para modelos que no soportan tool calling nativo en Ollama.
        Inyecta las tools como texto en el system prompt y parsea la respuesta JSON.
        """
        # Construir descripción de tools para el prompt
        tools_desc = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_lines = []
            for pname, pinfo in params.items():
                req_mark = " (requerido)" if pname in required else ""
                param_lines.append(f"    - {pname}: {pinfo.get('description', pinfo.get('type', 'string'))}{req_mark}")
            tools_desc.append(f"  {name}: {desc}\n" + "\n".join(param_lines))

        tools_text = "\n".join(tools_desc)

        augmented_prompt = (
            f"{system_prompt}\n\n"
            f"## Herramientas disponibles\n{tools_text}\n\n"
            "## Instrucciones de tool calling\n"
            "Para usar una herramienta, responde ÚNICAMENTE con un bloque JSON así:\n"
            "```json\n"
            '{"tool_call": {"name": "NOMBRE_TOOL", "arguments": {"param1": "valor1"}}}\n'
            "```\n"
            "NO escribas ningún otro texto fuera del bloque JSON si necesitas usar una herramienta.\n"
            "Si no necesitas usar herramientas, responde normalmente con texto."
        )

        # Reemplazar el system prompt en los mensajes
        fallback_messages = [{"role": "system", "content": augmented_prompt}]
        fallback_messages += original_messages[1:]  # Skip old system msg

        payload = {
            "model": self.model,
            "messages": fallback_messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(self._chat_url, json=payload)
                r.raise_for_status()
                data = r.json()

            message = data.get("message", {})
            content = (message.get("content") or "").strip()
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }

            # Intentar parsear tool call del contenido
            tool_calls = self._parse_tool_call_from_text(content)
            if tool_calls:
                return LLMResponse(
                    content="",
                    tool_calls=tool_calls,
                    finish_reason="tool_calls",
                    usage=usage,
                )

            return LLMResponse(
                content=content,
                tool_calls=[],
                finish_reason="stop",
                usage=usage,
            )
        except Exception as e:
            return LLMResponse(
                content=f"[Error en fallback tool calling: {e}]",
                finish_reason="error",
            )

    @staticmethod
    def _parse_tool_call_from_text(text: str) -> list[LLMToolCall]:
        """Intenta extraer un tool_call de texto libre del modelo."""
        import re as _re

        # Buscar bloque JSON con tool_call
        # Soporta ```json ... ``` o JSON directo
        patterns = [
            _re.compile(r'```json\s*\n?(.*?)\n?\s*```', _re.DOTALL),
            _re.compile(r'(\{[^{}]*"tool_call"[^{}]*\{.*?\}[^{}]*\})', _re.DOTALL),
        ]

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                try:
                    data = json.loads(match.group(1) if '```' in pattern.pattern else match.group(0))
                    tc = data.get("tool_call", {})
                    name = tc.get("name", "")
                    args = tc.get("arguments", {})
                    if name:
                        return [LLMToolCall(call_id="fallback_0", name=name, args=args)]
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Intentar parsear JSON puro sin wrapper
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                tc = data.get("tool_call", {})
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                if name:
                    return [LLMToolCall(call_id="fallback_0", name=name, args=args)]
        except json.JSONDecodeError:
            pass

        return []

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

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        system_prompt: str,
        tools: list[dict],
    ) -> LLMResponse:
        """Llama a la API OpenAI-compatible con tool calling nativo."""
        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages += [m.to_dict() for m in messages]

        payload: dict = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        timeout = httpx.Timeout(120.0, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(self._chat_url, json=payload)
                r.raise_for_status()
                data = r.json()

            choices = data.get("choices", [])
            if not choices:
                return LLMResponse(content="", finish_reason="error")

            choice = choices[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            finish_reason = choice.get("finish_reason", "stop")
            tool_calls: list[LLMToolCall] = []

            raw_tools = msg.get("tool_calls") or []
            if raw_tools:
                for tc in raw_tools:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            raw_args = {}
                    tool_calls.append(LLMToolCall(
                        call_id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        args=raw_args,
                    ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=data.get("usage", {}),
            )
        except Exception as e:
            return LLMResponse(
                content=f"[Error en tool calling: {e}]",
                finish_reason="error",
            )

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
