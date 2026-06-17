"""
iico_core/harness.py
====================
Orquestador central del sistema iico-agent.

El Harness es el único punto de contacto entre el UI y todo el núcleo.
El UI solo llama a `process_input()` y consume los `HarnessEvent` que devuelve.

Fase 2: implementa la arquitectura de memoria dual (Característica 3).
Fase 3: el Harness se convierte en una máquina de estados que orquesta el
flujo SDD completo: Definición → Planificación Autorizada → Ejecución Estricta.

Flujo de estados:
    IDLE → INTERVIEWING (detección SDD)
    INTERVIEWING → PLANNING (entrevista completa)
    PLANNING → AWAITING_APPROVAL (plan generado)
    AWAITING_APPROVAL → EXECUTING (usuario aprueba)
    EXECUTING → VERIFYING → IDLE (tareas terminadas)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from .bridge.shell import ShellBridge
from .index.splay_tree import SplayCacheMetrics, SplayTree
from .llm_client import LLMClient, create_client
from .memory.active import SkillRegistry
from .memory.passive import PassiveMemory
from .types import (
    AgentState,
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    ProviderConfig,
    SDDDocument,
    SkillDefinition,
    ToolResult,
)


class Harness:
    """
    Orquestador principal del iico-agent.

    Fase 2: Gestiona memoria dual, Skills y ShellBridge.
    Fase 3: Máquina de estados que orquesta el flujo SDD completo.
    """

    def __init__(self, config: HarnessConfig):
        self.config = config

        # --- LLM ---
        self.llm: LLMClient = create_client(
            provider_type=config.provider.type,
            endpoint=config.provider.endpoint,
            model=config.provider.model,
            temperature=config.provider.temperature,
        )

        # --- Memoria Pasiva (Nivel 1 fuente de verdad en Fase 1) ---
        self.passive_memory = PassiveMemory(config.memory_path)

        # --- Historial ---
        self.history: list[ChatMessage] = []

        # --- Splay Tree: Caché de Nivel 2 ---
        self._splay_metrics = SplayCacheMetrics()
        self._splay: SplayTree = SplayTree(
            max_nodes=config.splay_cache_size,
            metrics=self._splay_metrics,
        )

        # --- EmbeddingIndex: Nivel 1 semántico (opcional, requiere ONNX) ---
        self._embedding_index = None
        if config.use_embedding_search:
            self._init_embedding_index()

        # --- Fase 3: Máquina de Estados SDD ---
        self._state: AgentState = AgentState.IDLE
        self._project_root: Path | None = None
        self._sdd_manager = None
        self._task_manager = None
        self._react_loop = None

        # --- Skills (Fase 2) ---
        self._skill_registry: SkillRegistry | None = None
        self._bridge: ShellBridge | None = None
        if config.use_skills:
            self._skill_registry = SkillRegistry(config.skills_path)
            self._bridge = ShellBridge(
                default_timeout=config.skill_timeout,
                project_root=self._project_root,
            )

        # --- Aprobación de comandos (pausa hasta que el UI responda) ---
        self._approval_future: "asyncio.Future[bool] | None" = None

        if config.use_react_loop:
            self._init_reasoning_modules()

    def _init_reasoning_modules(self) -> None:
        """Inicializa los módulos de razonamiento de la Fase 3."""
        from .reasoning.sdd_manager import SDDManager
        from .reasoning.task_manager import TaskManager
        from .reasoning.react_loop import ReActLoop
        self._sdd_manager = SDDManager(self)
        self._task_manager = TaskManager(self)
        self._react_loop = ReActLoop(self)

    # ------------------------------------------------------------------
    # Aprobación de comandos de terminal (Fase 3)
    # ------------------------------------------------------------------

    def request_approval(self, command: str) -> None:
        """
        Prepara un Future de aprobación ANTES de yield COMMAND_APPROVAL_REQUIRED.
        Debe llamarse ANTES de hacer yield en el generador, de modo que cuando
        el generador se suspenda y el UI procese el evento, el Future ya exista
        y `approve()` / `reject()` puedan resolverlo.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        self._approval_future = loop.create_future()

    async def wait_for_approval(self) -> bool:
        """
        Espera a que el UI llame a approve() o reject().
        El Future ya fue creado en request_approval(), por lo que
        si el UI lo resolvió mientras el generador estaba suspendido
        (entre yield y este await), regresa inmediatamente.
        """
        if self._approval_future is None:
            return True
        result = await self._approval_future
        self._approval_future = None
        return result

    def approve(self) -> None:
        """Llamado por el UI: el usuario aprobó el comando."""
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(True)

    def reject(self) -> None:
        """Llamado por el UI: el usuario rechazó el comando."""
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(False)

    # ------------------------------------------------------------------
    # Inicialización del índice de embeddings
    # ------------------------------------------------------------------

    def _init_embedding_index(self) -> None:
        """Inicializa y construye el índice semántico con las notas actuales."""
        try:
            from .index.embedding import EmbeddingIndex
            self._embedding_index = EmbeddingIndex()
            notes = list(self.passive_memory)
            if notes:
                print(f"[Harness] Construyendo índice semántico ({len(notes)} notas)...")
                self._embedding_index.build_index(notes)
                print("[Harness] Índice semántico listo.")
        except ImportError as e:
            print(f"[Harness] Advertencia: búsqueda semántica desactivada. {e}")
            self._embedding_index = None

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    async def process_input(
        self,
        user_text: str,
    ) -> AsyncGenerator[HarnessEvent, None]:
        """
        Punto de entrada único para cualquier UI.
        En Fase 3 opera como una máquina de estados.
        """
        text = user_text.strip()
        if not text:
            return

        # Comandos slash (siempre tienen prioridad)
        if text.startswith("/"):
            async for event in self._handle_command(text):
                yield event
            return

        # ─────────────────────────────────────────────────────────────
        # Máquina de estados Fase 3
        # ─────────────────────────────────────────────────────────────

        if self._state == AgentState.INTERVIEWING and self._sdd_manager:
            # Opción de escape manual
            if text.lower() in ["cancelar", "salir", "exit", "stop", "abort", "abortar"]:
                self._state = AgentState.IDLE
                yield HarnessEvent(
                    type=HarnessEventType.STATE_CHANGED,
                    payload="🚫 Flujo de diseño SDD cancelado por el usuario.",
                )
                self.history.append(ChatMessage(role="system", content="[El usuario canceló el flujo SDD. Esperando nueva instrucción.]"))
                return

            # El usuario está respondiendo preguntas de la entrevista SDD
            async for event in self._sdd_manager.process_answer(text):
                yield event

            # Si la entrevista consolidó un SDD, generar el plan
            if self._sdd_manager.current_sdd is not None:
                self._state = AgentState.PLANNING
                yield HarnessEvent(
                    type=HarnessEventType.STATE_CHANGED,
                    payload="📋 Generando plan de acción...",
                )
                async for event in self._generate_and_propose_plan(
                    self._sdd_manager.current_sdd
                ):
                    yield event
            return

        if self._state == AgentState.AWAITING_APPROVAL and self._task_manager:
            # El usuario está aprobando o modificando el plan
            if self._task_manager.is_approval(text):
                self._state = AgentState.EXECUTING
                yield HarnessEvent(
                    type=HarnessEventType.STATE_CHANGED,
                    payload="⚙️ Ejecutando plan...",
                )
                async for event in self._execute_plan():
                    yield event
            else:
                # El usuario quiere cambios: volver a generar el plan
                self._state = AgentState.PLANNING
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="Entendido. Ajustando el plan con tus observaciones...",
                )
                # Tratar el texto como feedback e intentar re-planificar
                # Por ahora: respuesta simple y volver a proponer
                self.history.append(ChatMessage(role="user", content=text))
                async for event in self._generate_and_propose_plan(
                    self._sdd_manager.current_sdd if self._sdd_manager else None
                ):
                    yield event
            return

        # ─────────────────────────────────────────────────────────────
        # Detección de intención SDD (estado IDLE)
        # ─────────────────────────────────────────────────────────────

        if (
            self.config.use_react_loop
            and self._sdd_manager
            and self._sdd_manager.should_trigger(text)
        ):
            self._state = AgentState.INTERVIEWING
            self.history.append(ChatMessage(role="user", content=text))
            async for event in self._sdd_manager.start_interview(text):
                yield event
            return

        # ─────────────────────────────────────────────────────────────
        # ReAct directo (Fase 3, sin plan SDD)
        # ─────────────────────────────────────────────────────────────

        if self.config.use_react_loop and self._react_loop:
            self.history.append(ChatMessage(role="user", content=text))
            async for event in self._react_loop.execute_simple(text):
                yield event
            return

        # ─────────────────────────────────────────────────────────────
        # Fallback: chat normal (Fase 2)
        # ─────────────────────────────────────────────────────────────

        self.history.append(ChatMessage(role="user", content=text))
        system_prompt = self.build_system_prompt(query=text)

        full_response = ""
        try:
            async for token in self.llm.chat_stream(self.history, system_prompt):
                full_response += token
                yield HarnessEvent(type=HarnessEventType.TOKEN, payload=token)
        except Exception as e:
            yield HarnessEvent(type=HarnessEventType.ERROR, payload=str(e))
            self.history.pop()
            return

        self.history.append(ChatMessage(role="assistant", content=full_response))
        yield HarnessEvent(type=HarnessEventType.DONE, payload=full_response)

    # ------------------------------------------------------------------
    # System prompt dinámico — Arquitectura de Dos Niveles
    # ------------------------------------------------------------------

    def build_system_prompt(self, query: str = "") -> str:
        """
        Construye el system prompt inyectando solo el contexto relevante.

        Flujo (Característica 3 — Gestión de Contexto Híbrida):
        1. Consultar Splay Tree (Nivel 2) — O(1) para hits de localidad temporal
        2. Si miss → EmbeddingIndex (Nivel 1) — búsqueda semántica por coseno
        3. Si embeddings no disponibles → fallback a búsqueda por tags
        4. Insertar resultados nuevos en el Splay Tree
        5. Aplicar token budget y formatear
        """
        parts = [self.config.system_prompt_base]

        if not query or not self.config.use_passive_memory:
            if self.config.use_skills and self._skill_registry:
                skills_text = self._skill_registry.format_for_prompt()
                if skills_text:
                    parts.append(skills_text)
            return "\n\n".join(parts)

        # --- Paso 1: Consultar Splay Tree (Nivel 2) ---
        relevant_notes = []
        # peek_top retorna nodos sin modificar el árbol
        top_nodes = self._splay.peek_top(n=self.config.splay_peek_top)
        # Extraer los valores directamente de los nodos (sin llamar search() que splayea)
        cached_notes = [node.value for node in top_nodes]

        if cached_notes and self._splay_is_relevant(cached_notes, query):
            # Hit: el Splay resuelve sin vectorizar
            # Ahora sí llamamos search() para registrar el hit y splayear el nodo correcto
            best_key = top_nodes[0].key
            self._splay.search(best_key)  # registra hit en métricas
            relevant_notes = cached_notes
        else:
            # Miss: registrar en métricas
            if top_nodes:
                self._splay_metrics.record_access(depth=len(top_nodes), hit=False)

            # --- Paso 2: Fallback al Nivel 1 ---
            if self._embedding_index is not None and self.config.use_embedding_search:
                # Búsqueda semántica por embeddings
                results = self._embedding_index.search(
                    query,
                    threshold=self.config.embedding_threshold,
                    top_k=self.config.max_context_notes,
                )
                relevant_notes = [note for note, _ in results]
            else:
                # Fallback por tags (Fase 1 behavior)
                relevant_notes = self.passive_memory.get_relevant(
                    query,
                    method="tags",
                    max_results=self.config.max_context_notes,
                )

            # --- Paso 3: Insertar resultados en el Splay Tree ---
            for note in relevant_notes:
                self._splay.insert(note.id, note)

        # --- Paso 4: Aplicar token budget ---
        if relevant_notes:
            budgeted = self.passive_memory.apply_token_budget(
                relevant_notes,
                max_tokens=self.config.token_budget // 2,
            )
            context_text = self.passive_memory.format_for_prompt(budgeted)
            if context_text:
                parts.append(context_text)

        # --- Paso 5: Inyectar descripción de skills disponibles ---
        if self.config.use_skills and self._skill_registry:
            skills_text = self._skill_registry.format_for_prompt()
            if skills_text:
                parts.append(skills_text)

        return "\n\n".join(parts)

    def _splay_is_relevant(self, cached_notes: list, query: str) -> bool:
        """
        Heurística para decidir si los nodos cacheados son relevantes para el query.
        Compara palabras del query con los tags de las notas en caché.
        Si ninguna nota tiene tags que coincidan, hay divergencia semántica → miss.
        """
        import re, unicodedata

        def normalize(text: str) -> str:
            nfkd = unicodedata.normalize("NFKD", text.lower())
            return "".join(c for c in nfkd if not unicodedata.combining(c))

        query_words = set(re.findall(r"\b\w{3,}\b", normalize(query)))
        if not query_words:
            return bool(cached_notes)  # Sin palabras clave: asumir relevante

        for note in cached_notes:
            tag_set = {normalize(t) for t in getattr(note, "tags", [])}
            if query_words & tag_set:
                return True
        return False

    # ------------------------------------------------------------------
    # Ejecución de skills (para Fase 3 ReAct, ya disponible desde Fase 2)
    # ------------------------------------------------------------------

    def execute_skill(self, skill_name: str, args: dict) -> ToolResult | None:
        """
        Ejecuta una skill por nombre via ShellBridge.
        Retorna None si las skills no están habilitadas o la skill no existe.
        """
        if self._bridge is None or self._skill_registry is None:
            return None
        skill = self._skill_registry.get(skill_name)
        if skill is None:
            return ToolResult(
                skill_name=skill_name,
                output="",
                exit_code=1,
                error=f"Skill '{skill_name}' no encontrada en el registry.",
            )
        return self._bridge.execute(skill, args)

    # ------------------------------------------------------------------
    # Recarga de memoria
    # ------------------------------------------------------------------

    def reload_memory(self) -> None:
        """Recarga las notas desde disco y reconstruye el índice semántico."""
        self.passive_memory.reload()
        if self._embedding_index is not None:
            notes = list(self.passive_memory)
            self._embedding_index.build_index(notes)
        # Limpiar el Splay Tree para evitar datos obsoletos
        self._splay = SplayTree(
            max_nodes=self.config.splay_cache_size,
            metrics=self._splay_metrics,
        )

    # ------------------------------------------------------------------
    # Flujo SDD: generación de plan y ejecución (Fase 3)
    # ------------------------------------------------------------------

    async def _generate_and_propose_plan(
        self, sdd: SDDDocument | None
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Genera un plan desde el SDD y lo presenta al usuario para aprobación."""
        if sdd is None or self._task_manager is None:
            yield HarnessEvent(
                type=HarnessEventType.ERROR,
                payload="No hay SDD activo para generar un plan.",
            )
            return

        tasks, errors = await self._task_manager.generate_plan_from_sdd(sdd)

        if errors:
            error_text = "\n".join(errors)
            yield HarnessEvent(
                type=HarnessEventType.ERROR,
                payload=f"Error en el plan generado:\n{error_text}",
            )
            self._state = AgentState.IDLE
            return

        # Persistir las tareas como notas en el proyecto
        if self._project_root:
            self._task_manager.set_project_root(self._project_root)
            self._task_manager.save_tasks_as_notes()

        # Presentar el plan al usuario
        plan_text = self._task_manager.format_plan_for_display()
        self._state = AgentState.AWAITING_APPROVAL

        yield HarnessEvent(
            type=HarnessEventType.PLAN_PROPOSED,
            payload={
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "depends_on": t.depends_on,
                        "goals": [g.description for g in t.goals],
                    }
                    for t in tasks
                ]
            },
        )
        yield HarnessEvent(type=HarnessEventType.TOKEN, payload=plan_text)
        yield HarnessEvent(type=HarnessEventType.DONE, payload=plan_text)

    async def _execute_plan(self) -> AsyncGenerator[HarnessEvent, None]:
        """Ejecuta las tareas del plan una a una, validando dependencias."""
        if self._task_manager is None or self._react_loop is None:
            yield HarnessEvent(
                type=HarnessEventType.ERROR,
                payload="No hay plan activo para ejecutar.",
            )
            return

        sdd_tags = []
        if self._sdd_manager and self._sdd_manager.current_sdd:
            sdd_tags = self._sdd_manager.current_sdd.tags

        while True:
            task = self._task_manager.get_next_task()
            if task is None:
                break

            progress = self._task_manager.get_progress()
            yield HarnessEvent(
                type=HarnessEventType.STATE_CHANGED,
                payload=(
                    f"⚙️ Ejecutando {task.id} "
                    f"({progress['completed']+1}/{progress['total']}): "
                    f"{task.description}"
                ),
            )

            self._state = AgentState.EXECUTING
            async for event in self._react_loop.execute_task(task, sdd_tags):
                yield event

            # Persist the task state change after execution
            self._task_manager.save_tasks_as_notes()

            if task.status.value == "failed":
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload=(
                        f"⚠️ La tarea '{task.id}' falló. El plan se ha detenido.\n"
                        f"Puedes usar /abort para cancelar o revisar el error."
                    ),
                )
                self._state = AgentState.IDLE
                return

        # Todas las tareas completadas
        progress = self._task_manager.get_progress()
        summary = (
            f"✅ Plan completado: {progress['completed']}/{progress['total']} tareas.\n"
            "El agente ha terminado la ejecución."
        )
        yield HarnessEvent(type=HarnessEventType.STATE_CHANGED, payload=summary)
        yield HarnessEvent(type=HarnessEventType.TOKEN, payload=summary)
        yield HarnessEvent(type=HarnessEventType.DONE, payload=summary)
        self._state = AgentState.IDLE

    # ------------------------------------------------------------------
    # Comandos slash
    # ------------------------------------------------------------------

    async def _handle_command(
        self, text: str
    ) -> AsyncGenerator[HarnessEvent, None]:
        """Procesa comandos slash como /model, /clear, /memory, /skills, etc."""
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
            splay_summary = self._splay_metrics.summary()
            if count == 0:
                msg = "La memoria pasiva está vacía. Agrega archivos .md a memory_store/."
            else:
                ids = ", ".join(n.id for n in self.passive_memory)
                msg = (
                    f"Memoria pasiva: {count} nota(s) cargada(s).\n"
                    f"Notas: {ids}\n"
                    f"Splay Cache — hits: {splay_summary['hits']} | "
                    f"misses: {splay_summary['misses']} | "
                    f"hit rate: {splay_summary['hit_rate']:.1%} | "
                    f"profundidad media: {splay_summary['avg_depth']:.1f}"
                )
            yield HarnessEvent(type=HarnessEventType.SYSTEM, payload=msg)

        elif cmd == "/memory-reload":
            self.reload_memory()
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload=f"Memoria recargada: {len(self.passive_memory)} nota(s). Splay Tree limpiado.",
            )

        elif cmd == "/skills":
            if self._skill_registry is None:
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="Skills desactivadas. Activa 'use_skills=True' en HarnessConfig.",
                )
            else:
                count = len(self._skill_registry)
                if count == 0:
                    msg = "No hay skills registradas en skills/_registry.yaml."
                else:
                    names = ", ".join(s.name for s in self._skill_registry)
                    msg = f"Skills disponibles ({count}): {names}"
                yield HarnessEvent(type=HarnessEventType.SYSTEM, payload=msg)

        elif cmd == "/splay":
            summary = self._splay_metrics.summary()
            msg = (
                f"Splay Tree — tamaño: {self._splay.size}/{self.config.splay_cache_size} nodos\n"
                f"Hits: {summary['hits']} | Misses: {summary['misses']}\n"
                f"Hit rate: {summary['hit_rate']:.1%}\n"
                f"Profundidad promedio: {summary['avg_depth']:.2f} nodos"
            )
            yield HarnessEvent(type=HarnessEventType.SYSTEM, payload=msg)

        elif cmd == "/sdd":
            if not self.config.use_react_loop:
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="El flujo SDD requiere use_react_loop=True en la configuración.",
                )
            elif arg:
                # Iniciar SDD manualmente con descripción
                self._state = AgentState.INTERVIEWING
                async for event in self._sdd_manager.start_interview(arg):
                    yield event
            else:
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="Uso: /sdd <descripción del proyecto>\nO simplemente describe tu proyecto en lenguaje natural.",
                )

        elif cmd == "/plan":
            if self._task_manager and self._task_manager.tasks:
                plan_text = self._task_manager.format_plan_for_display()
                progress = self._task_manager.get_progress()
                status_line = (
                    f"Progreso: {progress['completed']}/{progress['total']} completadas | "
                    f"{progress['in_progress']} en progreso | "
                    f"{progress['failed']} fallidas\n\n"
                )
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload=status_line + plan_text,
                )
            else:
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="No hay ningún plan activo. Usa /sdd <proyecto> para iniciar uno.",
                )

        elif cmd == "/tasks":
            if self._task_manager and self._task_manager.tasks:
                lines = ["Estado de tareas:\n"]
                status_icons = {
                    "pending": "⏳",
                    "blocked": "🔒",
                    "in_progress": "⚙️",
                    "completed": "✅",
                    "failed": "❌",
                }
                for tid in self._task_manager.execution_order:
                    t = self._task_manager.tasks[tid]
                    icon = status_icons.get(t.status.value, "")
                    lines.append(f"{icon} [{t.id}] {t.description}")
                    if t.result_summary:
                        lines.append(f"   └ {t.result_summary[:80]}")
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="\n".join(lines),
                )
            else:
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload="No hay tareas en el plan actual.",
                )

        elif cmd == "/project":
            if arg:
                project_path = Path(arg).expanduser().resolve()
                if project_path.exists() and project_path.is_dir():
                    self._project_root = project_path
                    if self._sdd_manager:
                        self._sdd_manager.set_project_root(project_path)
                    if self._task_manager:
                        self._task_manager.set_project_root(project_path)
                    if self._bridge:
                        self._bridge.project_root = project_path
                    yield HarnessEvent(
                        type=HarnessEventType.SYSTEM,
                        payload=f"Carpeta raíz del proyecto: {project_path}",
                    )
                else:
                    yield HarnessEvent(
                        type=HarnessEventType.SYSTEM,
                        payload=f"La ruta '{arg}' no existe o no es un directorio.",
                    )
            else:
                current = str(self._project_root) if self._project_root else "(no configurada)"
                yield HarnessEvent(
                    type=HarnessEventType.SYSTEM,
                    payload=f"Carpeta raíz actual: {current}\nUso: /project <ruta>",
                )

        elif cmd == "/abort":
            self._state = AgentState.IDLE
            if self._task_manager:
                self._task_manager._tasks.clear()
                self._task_manager._execution_order.clear()
            if self._sdd_manager:
                self._sdd_manager._current_sdd = None
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload="⚠️ Flujo SDD cancelado. El agente vuelve al modo conversacional.",
            )

        else:
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload=(
                    f"Comando desconocido '{cmd}'.\n"
                    "Comandos disponibles: /clear, /memory, /memory-reload, /skills, "
                    "/splay, /sdd, /plan, /tasks, /project, /abort"
                ),
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

    @property
    def splay_metrics(self) -> SplayCacheMetrics:
        """Expone las métricas del Splay Tree para benchmarking."""
        return self._splay_metrics

    @property
    def skill_registry(self) -> SkillRegistry | None:
        return self._skill_registry
