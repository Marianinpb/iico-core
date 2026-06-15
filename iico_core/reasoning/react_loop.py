"""
iico_core/reasoning/react_loop.py
===================================
Bucle de razonamiento y acción (ReAct) para ejecución autónoma de tareas.

Flujo por iteración:
    1. THOUGHT  — el LLM analiza el estado actual con tool calling nativo
    2. ACTION   — si el LLM emite tool_calls, se ejecutan vía ShellBridge
    3. OBSERVATION — el resultado se inyecta como mensaje de herramienta
    4. Si no hay tool_calls → el LLM considera la tarea terminada

Auto-corrección integrada:
    Si una skill falla (exit_code != 0), se inyecta el error al LLM con un
    prompt de reflexión y se continúa el bucle (hasta MAX_RETRIES por paso).

Verificación de metas:
    Al finalizar cada TaskTemplate, se ejecutan los verification_skill de
    cada TaskGoal para confirmar que los criterios de aceptación se cumplieron.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, AsyncGenerator

from ..types import (
    ChatMessage,
    HarnessEvent,
    HarnessEventType,
    LLMToolCall,
    TaskGoal,
    TaskTemplate,
    TaskStatus,
)

if TYPE_CHECKING:
    from ..harness import Harness


class ReActLoop:
    """
    Bucle ReAct que ejecuta tareas usando las skills del SkillRegistry.

    Puede operar en dos modos:
    - execute_simple(): tarea conversacional directa (sin plan SDD)
    - execute_task(): tarea formal con metas comprobables y dependencias
    """

    MAX_ITERATIONS_PER_TASK = 12
    MAX_RETRIES_PER_STEP = 3

    def __init__(self, harness: "Harness"):
        self.harness = harness

    # ------------------------------------------------------------------
    # Modo simple: tarea directa sin plan SDD
    # ------------------------------------------------------------------

    async def execute_simple(
        self,
        user_text: str,
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Ejecuta una tarea directa sin plan SDD usando el bucle ReAct."""
        system_prompt = self.harness.build_system_prompt(query=user_text)
        tools = []
        if self.harness._skill_registry:
            tools = self.harness._skill_registry.get_tool_descriptions()
            if tools:
                system_prompt += (
                    "\n\nATENCIÓN: TIENES HERRAMIENTAS (TOOLS) DISPONIBLES. "
                    "DEBES usarlas (haciendo un tool call) para completar la tarea de forma autónoma "
                    "en lugar de decirle al usuario cómo hacerlo. No le pidas al usuario que ejecute comandos. "
                    "Ejecútalos tú mismo."
                )

        messages: list[ChatMessage] = list(self.harness.history)
        retry_count = 0

        for step in range(self.MAX_ITERATIONS_PER_TASK):
            yield HarnessEvent(
                type=HarnessEventType.THINKING,
                payload=f"Razonando (paso {step + 1})...",
            )

            response = await self.harness.llm.chat_with_tools(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            if response.finish_reason == "error":
                yield HarnessEvent(
                    type=HarnessEventType.ERROR,
                    payload=response.content,
                )
                return

            # Sin tool_calls → respuesta final
            if not response.tool_calls:
                self.harness.history.append(
                    ChatMessage(role="assistant", content=response.content)
                )
                yield HarnessEvent(type=HarnessEventType.TOKEN, payload=response.content)
                yield HarnessEvent(type=HarnessEventType.DONE, payload=response.content)
                return

            # Con tool_calls → ejecutar skills
            messages.append(ChatMessage(role="assistant", content=response.content or ""))

            for tc in response.tool_calls:
                yield HarnessEvent(type=HarnessEventType.SKILL_START, payload=tc.name)
                ok, retry_count = await self._execute_tool_call_async(tc, messages, retry_count)
                if not ok and retry_count >= self.MAX_RETRIES_PER_STEP:
                    yield HarnessEvent(
                        type=HarnessEventType.ERROR,
                        payload=f"La skill '{tc.name}' falló {self.MAX_RETRIES_PER_STEP} veces seguidas.",
                    )
                    return
                yield HarnessEvent(
                    type=HarnessEventType.SKILL_DONE,
                    payload={"skill": tc.name, "success": ok},
                )

        # Límite de iteraciones alcanzado
        yield HarnessEvent(
            type=HarnessEventType.SYSTEM,
            payload=f"[ReAct] Límite de {self.MAX_ITERATIONS_PER_TASK} pasos alcanzado.",
        )

    # ------------------------------------------------------------------
    # Modo tarea: ejecución formal con TaskTemplate
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        task: TaskTemplate,
        sdd_context_tags: list[str] | None = None,
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Ejecuta una tarea formal del plan SDD con verificación de metas."""
        task.status = TaskStatus.IN_PROGRESS
        yield HarnessEvent(
            type=HarnessEventType.TASK_STARTED,
            payload={"id": task.id, "description": task.description},
        )

        # Construir prompt específico para esta tarea
        system_prompt = self._build_task_prompt(task, sdd_context_tags or [])
        tools = []
        if self.harness._skill_registry:
            tools = self.harness._skill_registry.get_tool_descriptions()

        messages: list[ChatMessage] = [
            ChatMessage(role="user", content=task.description)
        ]
        retry_count = 0

        for step in range(self.MAX_ITERATIONS_PER_TASK):
            yield HarnessEvent(
                type=HarnessEventType.THINKING,
                payload=f"[{task.id}] Paso {step + 1}/{self.MAX_ITERATIONS_PER_TASK}",
            )

            response = await self.harness.llm.chat_with_tools(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            if response.finish_reason == "error":
                task.status = TaskStatus.FAILED
                yield HarnessEvent(
                    type=HarnessEventType.TASK_FAILED,
                    payload={"id": task.id, "error": response.content},
                )
                return

            # Sin tool_calls → tarea terminada según el LLM
            if not response.tool_calls:
                task.result_summary = response.content
                break

            messages.append(ChatMessage(role="assistant", content=response.content or ""))

            all_ok = True
            for tc in response.tool_calls:
                ok, retry_count = await self._execute_tool_call_async(
                    tc, messages, retry_count
                )
                if not ok and retry_count >= self.MAX_RETRIES_PER_STEP:
                    task.status = TaskStatus.FAILED
                    yield HarnessEvent(
                        type=HarnessEventType.TASK_FAILED,
                        payload={
                            "id": task.id,
                            "error": f"Skill '{tc.name}' falló {self.MAX_RETRIES_PER_STEP} veces.",
                        },
                    )
                    return
                if not ok:
                    all_ok = False

                yield HarnessEvent(
                    type=HarnessEventType.SKILL_DONE,
                    payload={"skill": tc.name, "success": ok},
                )

        # Verificar metas
        all_goals_met = True
        for goal in task.goals:
            async for ev in self._verify_goal(goal):
                yield ev
            if not goal.met:
                all_goals_met = False

        if all_goals_met:
            task.status = TaskStatus.COMPLETED
            yield HarnessEvent(
                type=HarnessEventType.TASK_COMPLETED,
                payload={"id": task.id, "summary": task.result_summary},
            )
        else:
            task.status = TaskStatus.FAILED
            failed_goals = [g.description for g in task.goals if not g.met]
            yield HarnessEvent(
                type=HarnessEventType.TASK_FAILED,
                payload={"id": task.id, "failed_goals": failed_goals},
            )

    # ------------------------------------------------------------------
    # Verificación de metas
    # ------------------------------------------------------------------

    async def _verify_goal(
        self, goal: TaskGoal
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Verifica una meta comprobable de una tarea."""
        if goal.verification_skill:
            result = self.harness.execute_skill(
                goal.verification_skill, goal.verification_args
            )
            goal.met = result is not None and result.success
        else:
            goal.met = True

        yield HarnessEvent(
            type=HarnessEventType.GOAL_VERIFIED,
            payload={"goal": goal.description, "met": goal.met},
        )

    # ------------------------------------------------------------------
    # Ejecución de tool calls con auto-corrección
    # ------------------------------------------------------------------

    async def _execute_tool_call_async(
        self,
        tc: LLMToolCall,
        messages: list[ChatMessage],
        retry_count: int,
    ) -> tuple[bool, int]:
        """
        Ejecuta un tool call vía ShellBridge y maneja auto-corrección.
        Retorna (success, retry_count_actualizado).
        """
        result = self.harness.execute_skill(tc.name, tc.args)

        if result is None:
            messages.append(ChatMessage(
                role="tool",
                content=json.dumps({
                    "error": f"Skill '{tc.name}' no encontrada en el registry.",
                    "available": [s.name for s in self.harness._skill_registry]
                    if self.harness._skill_registry else [],
                }),
            ))
            return False, retry_count + 1

        if result.success:
            messages.append(ChatMessage(
                role="tool",
                content=result.output or json.dumps({"status": "ok"}),
            ))
            return True, 0  # Reset reintentos

        # Fallo: inyectar error para auto-corrección
        retry_count += 1
        messages.append(ChatMessage(
            role="system",
            content=(
                f"La acción '{tc.name}' falló con el siguiente error:\n"
                f"```\n{result.error}\n```\n"
                "Analiza qué salió mal. Puedes:\n"
                "1. Corregir los argumentos y llamar a la misma skill.\n"
                "2. Usar una skill diferente que logre el mismo objetivo.\n"
                "3. Si el error es irrecuperable, explica el problema."
            ),
        ))
        return False, retry_count

    def _build_task_prompt(
        self,
        task: TaskTemplate,
        sdd_context_tags: list[str],
    ) -> str:
        """
        Construye el system prompt específico para una tarea.
        Consulta el SDD por tags (determinista) en lugar de inyectarlo completo.
        """
        base = self.harness.config.system_prompt_base

        # Obtener contexto del SDD por tags (búsqueda determinista)
        sdd_notes = []
        if self.harness.passive_memory and sdd_context_tags:
            sdd_notes = self.harness.passive_memory.get_relevant(
                " ".join(sdd_context_tags),
                method="tags",
                max_results=2,
            )

        # Obtener contexto de la tarea específica por su ID
        task_notes = self.harness.passive_memory.get_relevant(
            task.id,
            method="tags",
            max_results=1,
        ) if self.harness.passive_memory else []

        context_parts = [base]

        if sdd_notes or task_notes:
            context_parts.append("## Contexto del Proyecto\n")
            all_notes = sdd_notes + task_notes
            for note in all_notes:
                context_parts.append(f"### {note.id}\n{note.content[:800]}\n")

        # Descripción de la tarea actual
        goals_text = "\n".join(
            f"  - {g.description}" for g in task.goals
        ) or "  - (Completar la tarea de forma satisfactoria)"

        context_parts.append(
            f"## Tarea Actual: {task.id}\n"
            f"**Descripción:** {task.description}\n\n"
            f"**Metas a cumplir:**\n{goals_text}\n\n"
            "ATENCIÓN: TIENES HERRAMIENTAS (TOOLS) DISPONIBLES.\n"
            "DEBES usarlas (haciendo un tool call) para completar la tarea de forma autónoma "
            "en lugar de decirle al usuario cómo hacerlo. No le pidas al usuario que ejecute comandos. "
            "Ejecútalos tú mismo. Cuando hayas terminado, responde con un resumen de lo que hiciste."
        )

        # Skills disponibles
        if self.harness._skill_registry:
            skills_text = self.harness._skill_registry.format_for_prompt()
            if skills_text:
                context_parts.append(skills_text)

        return "\n\n".join(context_parts)
