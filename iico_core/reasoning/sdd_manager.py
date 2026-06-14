"""
iico_core/reasoning/sdd_manager.py
====================================
Gestor del flujo Spec-Driven Development (SDD).

Responsabilidades:
1. Detectar intención compleja en el mensaje del usuario (heurísticas)
2. Conducir una entrevista colaborativa para recopilar requisitos
3. Consolidar las respuestas en un SDDDocument estructurado
4. Persistir el SDD como nota Markdown+YAML en {proyecto}/.iico/sdd/

Diseño de la entrevista:
- El LLM analiza la solicitud inicial e identifica huecos en la especificación
- Genera hasta 5 preguntas concretas para el usuario
- El usuario responde y el LLM decide si necesita más info o puede consolidar
- Al consolidar, genera el SDDDocument + el TaskManager crea el plan

Eficiencia de contexto:
- El SDD se guarda como nota con tags [sdd, {título_slug}]
- El ReAct loop lo recupera por tags (O(n) determinista, sin vectores)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterator

import frontmatter

from ..types import (
    AgentState,
    ChatMessage,
    HarnessEvent,
    HarnessEventType,
    SDDDocument,
)

if TYPE_CHECKING:
    from ..harness import Harness


# Verbos que sugieren intención de creación/desarrollo complejo
_TRIGGER_VERBS = {
    "crea", "crear", "desarrolla", "desarrollar", "diseña", "diseñar",
    "implementa", "implementar", "construye", "construir", "programa",
    "programar", "genera", "generar", "automatiza", "automatizar",
    "refactoriza", "refactorizar", "build", "create", "develop", "design",
    "implement", "generate",
}

# Longitud mínima del mensaje para considerar flujo SDD
_MIN_TRIGGER_LENGTH = 40


class SDDManager:
    """
    Gestiona el ciclo de vida de las especificaciones (SDD).

    Estado interno durante la entrevista:
    - _original_request: solicitud inicial del usuario
    - _interview_qa: lista de (pregunta, respuesta) de la entrevista
    - _questions_pending: preguntas pendientes de respuesta del usuario
    """

    def __init__(self, harness: "Harness"):
        self.harness = harness
        self._original_request: str = ""
        self._interview_qa: list[tuple[str, str]] = []
        self._questions_pending: list[str] = []
        self._current_sdd: SDDDocument | None = None
        self._project_root: Path | None = None

    # ------------------------------------------------------------------
    # Configuración del proyecto
    # ------------------------------------------------------------------

    def set_project_root(self, root: Path) -> None:
        """Establece la carpeta raíz del proyecto activo."""
        self._project_root = root
        sdd_dir = root / ".iico" / "sdd"
        sdd_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Detección de intención
    # ------------------------------------------------------------------

    def should_trigger(self, user_text: str) -> bool:
        """
        Heurística: ¿el mensaje sugiere un flujo SDD?

        Criterios de activación (todos deben cumplirse):
        - Contiene verbos de creación/desarrollo
        - Longitud > _MIN_TRIGGER_LENGTH caracteres
        - No termina en "?" (no es una pregunta)
        """
        text = user_text.strip()
        if len(text) < _MIN_TRIGGER_LENGTH:
            return False
        if text.endswith("?"):
            return False

        words = set(re.findall(r"\b\w+\b", text.lower()))
        return bool(words & _TRIGGER_VERBS)

    # ------------------------------------------------------------------
    # Inicio de la entrevista
    # ------------------------------------------------------------------

    async def start_interview(
        self, user_text: str
    ) -> AsyncGenerator[HarnessEvent, None]:
        """
        Inicia la entrevista colaborativa.
        Pide al LLM identificar huecos en la solicitud y formular preguntas.
        """
        self._original_request = user_text
        self._interview_qa = []
        self._questions_pending = []

        # Notificar al UI que empezamos el flujo SDD
        yield HarnessEvent(
            type=HarnessEventType.STATE_CHANGED,
            payload="🔍 Iniciando flujo de diseño estructurado (SDD)...",
        )

        # Pedir al LLM que identifique huecos y genere preguntas
        questions = await self._generate_questions(user_text)
        self._questions_pending = questions

        if not questions:
            # El LLM tiene suficiente info para ir directo a consolidar
            yield HarnessEvent(
                type=HarnessEventType.SDD_STARTED,
                payload="Tengo suficiente información para comenzar.",
            )
            return

        # Presentar preguntas al usuario
        questions_text = self._format_questions(questions)
        yield HarnessEvent(
            type=HarnessEventType.SDD_QUESTION,
            payload=questions_text,
        )
        yield HarnessEvent(
            type=HarnessEventType.TOKEN,
            payload=questions_text,
        )
        yield HarnessEvent(
            type=HarnessEventType.DONE,
            payload=questions_text,
        )

    async def process_answer(
        self, answer: str
    ) -> AsyncGenerator[HarnessEvent, None]:
        """
        Procesa una respuesta del usuario durante la entrevista.
        Decide si necesita más información o puede consolidar el SDD.
        """
        # Guardar la respuesta asociada a las preguntas pendientes
        questions_str = " | ".join(self._questions_pending)
        self._interview_qa.append((questions_str, answer))
        self._questions_pending = []

        # Preguntar al LLM si necesita más info
        needs_more, new_questions = await self._check_completeness(answer)

        if needs_more and new_questions:
            self._questions_pending = new_questions
            questions_text = self._format_questions(new_questions)
            yield HarnessEvent(
                type=HarnessEventType.SDD_QUESTION,
                payload=questions_text,
            )
            yield HarnessEvent(type=HarnessEventType.TOKEN, payload=questions_text)
            yield HarnessEvent(type=HarnessEventType.DONE, payload=questions_text)
        else:
            # Consolidar el SDD
            yield HarnessEvent(
                type=HarnessEventType.STATE_CHANGED,
                payload="📋 Consolidando especificación...",
            )
            async for event in self._consolidate_sdd():
                yield event

    # ------------------------------------------------------------------
    # Lógica interna de la entrevista
    # ------------------------------------------------------------------

    async def _generate_questions(self, request: str) -> list[str]:
        """Pide al LLM formular preguntas sobre la solicitud del usuario."""
        prompt = (
            f"El usuario quiere: \"{request}\"\n\n"
            "Identifica hasta 4 aspectos técnicos o de diseño que no están claros "
            "y que necesitas conocer para crear un plan de implementación detallado. "
            "Si la solicitud ya es completamente clara, responde con una lista vacía.\n\n"
            "Responde SOLO con un JSON array de strings. Ejemplo:\n"
            '[\"¿Qué lenguaje de programación prefieres?\", \"¿Tienes alguna preferencia de formato de salida?\"]\n'
            "Si no necesitas más información: []"
        )

        response = await self.harness.llm.chat_with_tools(
            messages=[ChatMessage(role="user", content=prompt)],
            system_prompt=(
                "Eres un analista de requisitos técnicos. "
                "Responde únicamente con un JSON array de strings."
            ),
            tools=[],
        )

        return self._parse_questions_json(response.content)

    async def _check_completeness(
        self, latest_answer: str
    ) -> tuple[bool, list[str]]:
        """Determina si necesita más información tras la última respuesta."""
        context = self._build_interview_context()
        prompt = (
            f"Contexto de la entrevista hasta ahora:\n{context}\n\n"
            f"Última respuesta del usuario: \"{latest_answer}\"\n\n"
            "¿Necesitas más información para crear un plan detallado? "
            "Si sí, formula máximo 3 preguntas adicionales. "
            "Si no, responde con lista vacía.\n\n"
            "Responde SOLO con JSON: "
            '{"needs_more": true/false, "questions": [...]}'
        )

        response = await self.harness.llm.chat_with_tools(
            messages=[ChatMessage(role="user", content=prompt)],
            system_prompt=(
                "Eres un analista de requisitos. "
                "Responde solo con JSON válido."
            ),
            tools=[],
        )

        try:
            import json
            content = response.content.strip()
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            data = json.loads(content)
            needs_more = bool(data.get("needs_more", False))
            questions = [str(q) for q in data.get("questions", [])]
            return needs_more, questions
        except Exception:
            return False, []

    async def _consolidate_sdd(self) -> AsyncGenerator[HarnessEvent, None]:
        """Genera el SDDDocument a partir de la entrevista completa."""
        context = self._build_interview_context()
        prompt = (
            f"Solicitud original: \"{self._original_request}\"\n\n"
            f"Información recopilada en la entrevista:\n{context}\n\n"
            "Genera un documento de especificación técnica en JSON con este formato:\n"
            "```json\n"
            "{\n"
            '  "title": "Título del proyecto",\n'
            '  "description": "Descripción técnica concisa",\n'
            '  "requirements": ["Requisito 1", "Requisito 2"],\n'
            '  "constraints": ["Restricción 1"]\n'
            "}\n"
            "```\n"
            "Responde SOLO con el JSON."
        )

        response = await self.harness.llm.chat_with_tools(
            messages=[ChatMessage(role="user", content=prompt)],
            system_prompt=(
                "Eres un arquitecto de software. "
                "Responde únicamente con JSON válido."
            ),
            tools=[],
        )

        sdd = self._parse_sdd_json(response.content)
        self._current_sdd = sdd

        # Persistir como nota consultable
        if self._project_root:
            self.save_sdd_as_note(sdd)

        yield HarnessEvent(
            type=HarnessEventType.SDD_STARTED,
            payload={
                "title": sdd.title,
                "description": sdd.description,
                "requirements": sdd.requirements,
            },
        )
        yield HarnessEvent(
            type=HarnessEventType.STATE_CHANGED,
            payload=f"📄 SDD generado: {sdd.title}",
        )

    def _build_interview_context(self) -> str:
        """Construye el contexto de la entrevista como texto legible."""
        if not self._interview_qa:
            return "(sin entrevista previa)"
        parts = []
        for i, (questions, answer) in enumerate(self._interview_qa, 1):
            parts.append(f"Ronda {i}:\nPreguntas: {questions}\nRespuesta: {answer}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_questions_json(self, raw: str) -> list[str]:
        """Parsea la lista de preguntas del LLM."""
        import json
        try:
            content = raw.strip()
            # Buscar el array JSON si viene con texto adicional
            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1:
                return []
            content = content[start:end]
            questions = json.loads(content)
            return [str(q) for q in questions if q][:4]  # Máx. 4 preguntas
        except Exception:
            return []

    def _parse_sdd_json(self, raw: str) -> SDDDocument:
        """Parsea el SDDDocument del JSON generado por el LLM."""
        import json
        try:
            content = raw.strip()
            if "```" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end]
            data = json.loads(content)
        except Exception:
            data = {}

        title = str(data.get("title", "Proyecto Sin Título"))
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:30]

        sdd = SDDDocument(
            title=title,
            description=str(data.get("description", self._original_request)),
            requirements=[str(r) for r in data.get("requirements", [])],
            constraints=[str(c) for c in data.get("constraints", [])],
            tags=["sdd", slug],
        )

        # Construir el markdown del SDD
        reqs = "\n".join(f"- {r}" for r in sdd.requirements)
        constraints = "\n".join(f"- {c}" for c in sdd.constraints)
        sdd.raw_markdown = (
            f"# {sdd.title}\n\n"
            f"**Descripción:** {sdd.description}\n\n"
            f"## Requisitos\n{reqs or '(ninguno especificado)'}\n\n"
            f"## Restricciones\n{constraints or '(ninguna especificada)'}\n\n"
            f"## Solicitud Original\n{self._original_request}"
        )

        return sdd

    def _format_questions(self, questions: list[str]) -> str:
        """Formatea las preguntas para presentarlas al usuario."""
        lines = ["Necesito algunas aclaraciones antes de crear el plan:\n"]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("\nPor favor responde todas las preguntas anteriores.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save_sdd_as_note(self, sdd: SDDDocument) -> None:
        """
        Persiste el SDD como nota Markdown+YAML en {proyecto}/.iico/sdd/.
        Tags deterministas para búsqueda sin semántica.
        """
        if not self._project_root:
            return
        sdd_dir = self._project_root / ".iico" / "sdd"
        sdd_dir.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^a-z0-9]+", "_", sdd.title.lower())[:30]
        note_id = f"sdd_{slug}"

        post = frontmatter.Post(
            sdd.raw_markdown,
            id=note_id,
            tags=sdd.tags + ["sdd"],
            priority=8,
        )
        path = sdd_dir / f"{note_id}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        # Registrar en la memoria pasiva para que el Harness lo encuentre
        self.harness.passive_memory.add_note(
            note_id=note_id,
            content=sdd.raw_markdown,
            tags=sdd.tags + ["sdd"],
            priority=8,
        )

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def current_sdd(self) -> SDDDocument | None:
        return self._current_sdd

    @property
    def has_pending_questions(self) -> bool:
        return bool(self._questions_pending)
