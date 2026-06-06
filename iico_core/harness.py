"""
iico_core/harness.py
====================
Orquestador central del sistema iico-agent.

El Harness es el único punto de contacto entre el UI y todo el núcleo.
El UI solo llama a `process_input()` y consume los `HarnessEvent` que devuelve.

Fase 2: implementa la arquitectura de memoria dual (Característica 3):
    Nivel 1: EmbeddingIndex  — búsqueda semántica (fuente de verdad)
    Nivel 2: SplayTree       — caché de trabajo rápida (localidad temporal)

Flujo de build_system_prompt():
    1. Consultar raíz/hijos del Splay (Nivel 2) → hit? → usar sin vectorizar
    2. Miss → EmbeddingIndex.search() (Nivel 1)
    3. Insertar resultados del Nivel 1 en el Splay
    4. Aplicar token budget
    5. Inyectar notas + firmas de skills al system prompt
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
    ChatMessage,
    HarnessConfig,
    HarnessEvent,
    HarnessEventType,
    ProviderConfig,
    SkillDefinition,
    ToolResult,
)


class Harness:
    """
    Orquestador principal del iico-agent.

    Responsabilidades en Fase 2:
    - Gestionar el historial de mensajes
    - Construir el system prompt dinámico con arquitectura de dos niveles
    - Gestionar el Splay Tree como caché de contexto
    - Llamar al LLM y emitir HarnessEvents
    - Mantener el SkillRegistry y ShellBridge disponibles
    - Manejar comandos slash (/)
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

        # --- Skills (Fase 2) ---
        self._skill_registry: SkillRegistry | None = None
        self._bridge: ShellBridge | None = None
        if config.use_skills:
            self._skill_registry = SkillRegistry(config.skills_path)
            self._bridge = ShellBridge(default_timeout=config.skill_timeout)

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
            self.history.pop()   # Revertir el mensaje si falló
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

        else:
            yield HarnessEvent(
                type=HarnessEventType.SYSTEM,
                payload=(
                    f"Comando desconocido '{cmd}'.\n"
                    "Comandos disponibles: /clear, /memory, /memory-reload, /skills, /splay"
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
