"""
iico_core/reasoning/task_manager.py
=====================================
Gestor de tareas para el flujo SDD.

Responsabilidades:
- Parsear el plan JSON generado por el LLM y crear TaskTemplates
- Validar el grafo de dependencias (detectar ciclos con Kahn's algorithm)
- Determinar cuál es la siguiente tarea ejecutable
- Persistir las tareas como notas Markdown+YAML en {proyecto}/.iico/tasks/
  con tags deterministas (ej: [plan, task_1, subtask_a]) para que la
  búsqueda por tags del PassiveMemory las encuentre sin búsqueda semántica
- Actualizar el estado de las tareas y calcular el progreso
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from ..types import (
    AgentState,
    HarnessEvent,
    HarnessEventType,
    SDDDocument,
    TaskGoal,
    TaskStatus,
    TaskTemplate,
)

if TYPE_CHECKING:
    from ..harness import Harness


# Palabras clave que el usuario puede decir para aprobar el plan
_APPROVAL_WORDS = {
    "sí", "si", "ok", "apruebo", "adelante", "yes", "confirmo",
    "procede", "ejecuta", "listo", "dale", "bien", "correcto",
}


class TaskManager:
    """
    Gestiona el ciclo de vida del plan de tareas dentro de un proyecto.

    El plan se almacena en {project_root}/.iico/tasks/ como notas Markdown+YAML
    consultables por el PassiveMemory usando tags deterministas.
    """

    def __init__(self, harness: "Harness"):
        self.harness = harness
        self._tasks: dict[str, TaskTemplate] = {}   # id → task
        self._execution_order: list[str] = []        # Orden topológico
        self._sdd: SDDDocument | None = None
        self._project_root: Path | None = None

    # ------------------------------------------------------------------
    # Configuración del proyecto
    # ------------------------------------------------------------------

    def set_project_root(self, root: Path) -> None:
        """Establece la carpeta raíz del proyecto activo."""
        self._project_root = root
        tasks_dir = root / ".iico" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def iico_dir(self) -> Path | None:
        return self._project_root / ".iico" if self._project_root else None

    # ------------------------------------------------------------------
    # Generación del plan desde el LLM
    # ------------------------------------------------------------------

    async def generate_plan_from_sdd(
        self, sdd: SDDDocument
    ) -> tuple[list[TaskTemplate], list[str]]:
        """
        Pide al LLM generar un plan de tareas basado en el SDD.
        Retorna (tasks, errors).
        """
        self._sdd = sdd

        plan_prompt = (
            f"Eres un planificador de proyectos de software.\n\n"
            f"Basándote en el siguiente documento de especificación:\n\n"
            f"**Título:** {sdd.title}\n"
            f"**Descripción:** {sdd.description}\n"
            f"**Requisitos:**\n" + "\n".join(f"- {r}" for r in sdd.requirements) +
            f"\n**Restricciones:**\n" + "\n".join(f"- {c}" for c in sdd.constraints) +
            "\n\nGenera un plan de tareas en formato JSON con esta estructura exacta:\n"
            "```json\n"
            "{\n"
            '  "tasks": [\n'
            "    {\n"
            '      "id": "task_1",\n'
            '      "description": "Descripción clara de la tarea",\n'
            '      "goals": [\n'
            '        {"description": "Meta comprobable 1", "verification_skill": null},\n'
            '        {"description": "Meta comprobable 2", "verification_skill": "list_files"}\n'
            "      ],\n"
            '      "depends_on": []\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Reglas:\n"
            "- Máximo 8 tareas.\n"
            "- Las dependencias (depends_on) deben ser IDs de tareas anteriores.\n"
            "- Las metas deben ser verificables y concretas.\n"
            "- verification_skill puede ser: list_files, read_file_snippet, "
            "read_csv_head, compile_latex, compile_mermaid, o null.\n"
            "- Responde SOLO con el JSON, sin texto adicional."
        )

        from ..types import ChatMessage
        response = await self.harness.llm.chat_with_tools(
            messages=[ChatMessage(role="user", content=plan_prompt)],
            system_prompt="Eres un planificador técnico. Responde únicamente con JSON válido.",
            tools=[],  # Sin tools para esta llamada
        )

        return self._parse_plan_json(response.content)

    def _parse_plan_json(
        self, raw: str
    ) -> tuple[list[TaskTemplate], list[str]]:
        """Parsea el JSON del plan generado por el LLM."""
        errors: list[str] = []

        # Extraer bloque JSON (el LLM a veces añade ```json ... ```)
        content = raw.strip()
        if "```" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            content = content[start:end] if start != -1 else content

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"El LLM no generó JSON válido: {e}")
            return [], errors

        tasks_raw = data.get("tasks", [])
        tasks: list[TaskTemplate] = []

        for t in tasks_raw:
            goals = [
                TaskGoal(
                    description=g.get("description", ""),
                    verification_skill=g.get("verification_skill"),
                )
                for g in t.get("goals", [])
            ]
            task_id = str(t.get("id", f"task_{len(tasks)+1}"))
            tasks.append(TaskTemplate(
                id=task_id,
                description=str(t.get("description", "")),
                goals=goals,
                depends_on=[str(d) for d in t.get("depends_on", [])],
                tags=["plan", task_id],
            ))

        self._tasks = {t.id: t for t in tasks}
        dep_errors = self.validate_dependencies()
        errors.extend(dep_errors)

        if not dep_errors:
            self._compute_execution_order()

        return tasks, errors

    # ------------------------------------------------------------------
    # Validación de dependencias (Kahn's algorithm)
    # ------------------------------------------------------------------

    def validate_dependencies(self) -> list[str]:
        """
        Valida el grafo de dependencias.
        Detecta dependencias inexistentes y ciclos.
        Retorna lista de errores (vacía si todo está OK).
        """
        errors: list[str] = []
        ids = set(self._tasks.keys())

        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in ids:
                    errors.append(
                        f"Task '{task.id}' depende de '{dep}' que no existe."
                    )

        if errors:
            return errors

        # Detección de ciclos con Kahn
        in_degree = {t: 0 for t in ids}
        for task in self._tasks.values():
            for dep in task.depends_on:
                in_degree[task.id] = in_degree.get(task.id, 0) + 1

        queue = deque(t for t, d in in_degree.items() if d == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for task in self._tasks.values():
                if node in task.depends_on:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)

        if visited != len(ids):
            errors.append("El grafo de dependencias tiene ciclos. Revisa las tareas.")

        return errors

    def _compute_execution_order(self) -> None:
        """Calcula el orden topológico de ejecución."""
        in_degree = {t: 0 for t in self._tasks}
        for task in self._tasks.values():
            for _ in task.depends_on:
                in_degree[task.id] += 1

        queue = deque(
            tid for tid, d in sorted(in_degree.items(), key=lambda x: x[0])
            if d == 0
        )
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for task in self._tasks.values():
                if node in task.depends_on:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)

        self._execution_order = order

    # ------------------------------------------------------------------
    # Progreso y estado
    # ------------------------------------------------------------------

    def get_next_task(self) -> TaskTemplate | None:
        """Retorna la siguiente tarea cuyas dependencias están satisfechas."""
        completed = {
            tid for tid, t in self._tasks.items()
            if t.status == TaskStatus.COMPLETED
        }
        for task_id in self._execution_order:
            task = self._tasks[task_id]
            if task.status == TaskStatus.PENDING and task.is_ready(completed):
                return task
        return None

    def mark_completed(self, task_id: str, summary: str = "") -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.COMPLETED
            self._tasks[task_id].result_summary = summary

    def mark_failed(self, task_id: str, error: str = "") -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.FAILED
            self._tasks[task_id].result_summary = f"[FAILED] {error}"

    def get_progress(self) -> dict:
        """Resumen de progreso para el UI."""
        total = len(self._tasks)
        counts = {s: 0 for s in TaskStatus}
        for t in self._tasks.values():
            counts[t.status] += 1
        return {
            "total": total,
            "completed": counts[TaskStatus.COMPLETED],
            "failed": counts[TaskStatus.FAILED],
            "in_progress": counts[TaskStatus.IN_PROGRESS],
            "pending": counts[TaskStatus.PENDING],
        }

    def is_plan_done(self) -> bool:
        """¿Todas las tareas están completadas?"""
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def has_failures(self) -> bool:
        """¿Alguna tarea falló?"""
        return any(t.status == TaskStatus.FAILED for t in self._tasks.values())

    def format_plan_for_display(self) -> str:
        """Formatea el plan para mostrar al usuario antes de la aprobación."""
        if not self._tasks:
            return "Sin tareas."
        lines = ["## Plan de Acción\n"]
        for task_id in self._execution_order:
            task = self._tasks[task_id]
            deps = ", ".join(task.depends_on) if task.depends_on else "ninguna"
            goals_text = "\n".join(f"    ✓ {g.description}" for g in task.goals)
            lines.append(
                f"**{task.id}**: {task.description}\n"
                f"  Dependencias: {deps}\n"
                f"  Metas:\n{goals_text}\n"
            )
        lines.append(
            "\n¿Apruebas este plan? Responde 'sí' para comenzar la ejecución, "
            "o describe los cambios que deseas hacer."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistencia en el proyecto
    # ------------------------------------------------------------------

    def save_tasks_as_notes(self) -> None:
        """
        Persiste cada tarea como nota Markdown+YAML en {proyecto}/.iico/tasks/.
        Usa tags deterministas para búsqueda eficiente sin semántica.
        """
        if not self._project_root:
            return
        tasks_dir = self._project_root / ".iico" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        for task in self._tasks.values():
            goals_md = "\n".join(
                f"- {g.description}"
                + (f" [verifica: `{g.verification_skill}`]" if g.verification_skill else "")
                for g in task.goals
            )
            content = (
                f"# {task.id}: {task.description}\n\n"
                f"**Estado:** {task.status.value}\n"
                f"**Dependencias:** {', '.join(task.depends_on) or 'ninguna'}\n\n"
                f"## Metas\n{goals_md}\n\n"
                f"## Resultado\n{task.result_summary or '(pendiente)'}"
            )
            post = frontmatter.Post(
                content,
                id=task.id,
                tags=task.tags + ["plan"],
                priority=5,
                status=task.status.value,
                depends_on=task.depends_on,
            )
            path = tasks_dir / f"{task.id}.md"
            with open(path, "w", encoding="utf-8") as f:
                f.write(frontmatter.dumps(post))

    def is_approval(self, text: str) -> bool:
        """¿El texto del usuario es una aprobación del plan?"""
        return text.strip().lower() in _APPROVAL_WORDS

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def tasks(self) -> dict[str, TaskTemplate]:
        return self._tasks

    @property
    def execution_order(self) -> list[str]:
        return self._execution_order
