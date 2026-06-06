"""
iico_core/index/splay_tree.py
==============================
Árbol Splay como Caché de Nivel 2 (Característica 3).

ARQUITECTURA: Este árbol NO es el índice primario de notas/skills.
Es una caché de trabajo rápida que almacena SOLO nodos previamente
validados por el Nivel 1 (EmbeddingIndex). El flujo es:

    1. Consultar raíz/hijos del Splay → hit? → usar sin vectorizar
    2. Miss → delegar a EmbeddingIndex (Nivel 1)
    3. Resultado del Nivel 1 → insertar en el Splay
    4. Nodos obsoletos migran naturalmente hacia las hojas

Propiedades del Splay Tree:
- search(x): O(log n) amortizado — splayea x a la raíz
- insert(x): O(log n) amortizado
- delete(x): O(log n) amortizado
- La localidad temporal hace que accesos frecuentes sean O(1) en práctica

Para la tesis: SplayCacheMetrics registra hit rate y profundidad promedio,
lo que permite graficar la convergencia del árbol con el tiempo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Nodo del árbol
# ---------------------------------------------------------------------------

@dataclass
class SplayNode:
    """Nodo del Splay Tree."""
    key: str                     # ID de la nota o skill
    value: Any                   # PassiveNote o SkillDefinition
    left:  "SplayNode | None" = field(default=None, repr=False)
    right: "SplayNode | None" = field(default=None, repr=False)
    parent: "SplayNode | None" = field(default=None, repr=False)
    access_count: int = 0        # Cuántas veces fue accedido (para benchmarking)


# ---------------------------------------------------------------------------
# Métricas para la tesis
# ---------------------------------------------------------------------------

class SplayCacheMetrics:
    """
    Registra métricas del Splay Tree para el framework de evaluación.

    Métricas clave para la tesis:
    - hit_rate: % de consultas resueltas desde la caché
    - avg_depth: profundidad promedio de acceso (debe decrecer con el tiempo)
    - splay_convergence: observable al graficar avg_depth vs accesos_totales
    """

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.total_depth: int = 0
        self.total_accesses: int = 0
        # Historial para graficar convergencia
        self._depth_history: list[tuple[int, float]] = []  # (acceso_n, profundidad)

    def record_access(self, depth: int, hit: bool) -> None:
        """Registra un acceso al árbol."""
        self.total_accesses += 1
        self.total_depth += depth
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        # Guardar muestra cada 10 accesos (para historial de convergencia)
        if self.total_accesses % 10 == 0:
            self._depth_history.append((self.total_accesses, self.avg_depth))

    @property
    def hit_rate(self) -> float:
        """Porcentaje de hits (0.0 - 1.0)."""
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    @property
    def avg_depth(self) -> float:
        """Profundidad promedio de acceso."""
        if self.total_accesses == 0:
            return 0.0
        return self.total_depth / self.total_accesses

    @property
    def depth_history(self) -> list[tuple[int, float]]:
        """Historial de (n_accesos, profundidad_promedio) para graficar convergencia."""
        return list(self._depth_history)

    def summary(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_accesses": self.total_accesses,
            "hit_rate": round(self.hit_rate, 4),
            "avg_depth": round(self.avg_depth, 2),
        }

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.total_depth = 0
        self.total_accesses = 0
        self._depth_history.clear()


# ---------------------------------------------------------------------------
# Splay Tree
# ---------------------------------------------------------------------------

class SplayTree:
    """
    Árbol Splay auto-ajustable.

    Implementación pura en Python sin dependencias externas.
    Cada operación de búsqueda splayea el nodo encontrado a la raíz,
    lo que garantiza localidad temporal: los nodos más accedidos
    permanecen cerca de la raíz con costo de acceso ≈ O(1).
    """

    def __init__(self, max_nodes: int = 50, metrics: SplayCacheMetrics | None = None):
        """
        Args:
            max_nodes: capacidad máxima del caché. Cuando se supera, se
                       evictan los nodos más profundos (menos frecuentes).
            metrics: colector de métricas. Si es None, se crea uno interno.
        """
        self._root: SplayNode | None = None
        self._size: int = 0
        self.max_nodes = max_nodes
        self.metrics: SplayCacheMetrics = metrics or SplayCacheMetrics()

    # ------------------------------------------------------------------
    # Operaciones públicas
    # ------------------------------------------------------------------

    def search(self, key: str) -> SplayNode | None:
        """
        Busca un nodo por key. Si existe, lo splayea a la raíz.
        Registra la profundidad del acceso en las métricas.
        """
        node, depth = self._find(key)
        if node is not None:
            self._splay(node)
            node.access_count += 1
            self.metrics.record_access(depth, hit=True)
            return node
        else:
            self.metrics.record_access(depth, hit=False)
            return None

    def insert(self, key: str, value: Any) -> SplayNode:
        """
        Inserta un nodo o actualiza su valor si ya existe.
        El nodo insertado sube a la raíz.
        """
        existing, _ = self._find(key)
        if existing is not None:
            # Actualizar valor y splayear
            existing.value = value
            self._splay(existing)
            return existing

        node = SplayNode(key=key, value=value)
        if self._root is None:
            self._root = node
        else:
            # Insertar como nueva hoja en el lugar apropiado
            self._insert_node(node)
            self._splay(node)
        self._size += 1

        # Evictar si se supera la capacidad
        if self._size > self.max_nodes:
            self._evict_deepest()

        return node

    def delete(self, key: str) -> bool:
        """Elimina un nodo. Retorna True si existía."""
        node, _ = self._find(key)
        if node is None:
            return False
        self._splay(node)
        # Después del splay, node es la raíz
        # Unir los subárboles izquierdo y derecho
        left = self._root.left
        right = self._root.right
        if left:
            left.parent = None
        if right:
            right.parent = None

        if left is None:
            self._root = right
        elif right is None:
            self._root = left
        else:
            # Encontrar el máximo del subárbol izquierdo
            max_left = left
            while max_left.right:
                max_left = max_left.right
            self._splay_within(max_left, left)
            max_left.right = right
            right.parent = max_left
            max_left.parent = None
            self._root = max_left

        self._size -= 1
        return True

    def peek_top(self, n: int = 3) -> list[SplayNode]:
        """
        Devuelve los n nodos más cercanos a la raíz (raíz + hijos inmediatos).
        Usado por el Harness para el hit check rápido sin splayear.
        """
        result: list[SplayNode] = []
        self._collect_top(self._root, result, n)
        return result

    def peek_keys_top(self, n: int = 3) -> set[str]:
        """Versión rápida que solo retorna las keys de los nodos top."""
        return {node.key for node in self.peek_top(n)}

    # ------------------------------------------------------------------
    # Rotaciones (operaciones de splay)
    # ------------------------------------------------------------------

    def _splay(self, node: SplayNode) -> None:
        """Splayea el nodo hasta la raíz mediante rotaciones zig, zig-zig, zig-zag."""
        while node.parent is not None:
            parent = node.parent
            grandparent = parent.parent

            if grandparent is None:
                # Zig: el padre es la raíz
                if parent.left is node:
                    self._rotate_right(parent)
                else:
                    self._rotate_left(parent)
            elif parent.left is node and grandparent.left is parent:
                # Zig-zig (ambos izquierda)
                self._rotate_right(grandparent)
                self._rotate_right(parent)
            elif parent.right is node and grandparent.right is parent:
                # Zig-zig (ambos derecha)
                self._rotate_left(grandparent)
                self._rotate_left(parent)
            elif parent.left is node and grandparent.right is parent:
                # Zig-zag (izquierda-derecha)
                self._rotate_right(parent)
                self._rotate_left(grandparent)
            else:
                # Zig-zag (derecha-izquierda)
                self._rotate_left(parent)
                self._rotate_right(grandparent)

    def _splay_within(self, node: SplayNode, subtree_root: SplayNode) -> None:
        """Splayea node dentro del subárbol rooted en subtree_root."""
        while node.parent is not None and node.parent is not subtree_root.parent:
            self._splay(node)
            if self._root is node:
                break

    def _rotate_right(self, node: SplayNode) -> None:
        """Rotación derecha sobre node."""
        left_child = node.left
        if left_child is None:
            return

        node.left = left_child.right
        if left_child.right:
            left_child.right.parent = node

        left_child.parent = node.parent
        if node.parent is None:
            self._root = left_child
        elif node.parent.left is node:
            node.parent.left = left_child
        else:
            node.parent.right = left_child

        left_child.right = node
        node.parent = left_child

    def _rotate_left(self, node: SplayNode) -> None:
        """Rotación izquierda sobre node."""
        right_child = node.right
        if right_child is None:
            return

        node.right = right_child.left
        if right_child.left:
            right_child.left.parent = node

        right_child.parent = node.parent
        if node.parent is None:
            self._root = right_child
        elif node.parent.left is node:
            node.parent.left = right_child
        else:
            node.parent.right = right_child

        right_child.left = node
        node.parent = right_child

    # ------------------------------------------------------------------
    # Operaciones internas
    # ------------------------------------------------------------------

    def _find(self, key: str) -> tuple[SplayNode | None, int]:
        """Busca un nodo por key. Retorna (nodo, profundidad)."""
        current = self._root
        depth = 0
        while current:
            if key == current.key:
                return current, depth
            elif key < current.key:
                current = current.left
            else:
                current = current.right
            depth += 1
        return None, depth

    def _insert_node(self, node: SplayNode) -> None:
        """Inserta un nodo en el BST sin splayear."""
        current = self._root
        while True:
            if node.key < current.key:
                if current.left is None:
                    current.left = node
                    node.parent = current
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = node
                    node.parent = current
                    return
                current = current.right

    def _collect_top(
        self,
        node: SplayNode | None,
        result: list[SplayNode],
        n: int,
        depth: int = 0,
    ) -> None:
        """BFS para recolectar los nodos más cercanos a la raíz."""
        if node is None or len(result) >= n:
            return
        # Nivel por nivel: primero el nodo actual
        if depth <= 2:  # Raíz + 2 niveles = área de hit rápido
            result.append(node)
        if len(result) < n:
            self._collect_top(node.left, result, n, depth + 1)
        if len(result) < n:
            self._collect_top(node.right, result, n, depth + 1)

    def _evict_deepest(self) -> None:
        """Elimina el nodo más profundo (menos frecuentemente accedido)."""
        if self._root is None:
            return
        # Encontrar la hoja más profunda
        deepest = self._find_deepest(self._root)
        if deepest:
            self.delete(deepest.key)

    def _find_deepest(self, node: SplayNode | None) -> SplayNode | None:
        """Retorna la hoja más profunda del árbol."""
        if node is None:
            return None
        if node.left is None and node.right is None:
            return node
        left_deepest  = self._find_deepest(node.left)
        right_deepest = self._find_deepest(node.right)
        if left_deepest is None:
            return right_deepest
        if right_deepest is None:
            return left_deepest
        # Retornar la más profunda (heurística: la del subárbol más grande)
        left_depth  = self._node_depth(left_deepest)
        right_depth = self._node_depth(right_deepest)
        return left_deepest if left_depth >= right_depth else right_deepest

    def _node_depth(self, node: SplayNode) -> int:
        """Calcula la profundidad de un nodo desde la raíz."""
        depth = 0
        current = node
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth

    # ------------------------------------------------------------------
    # Propiedades e inspección
    # ------------------------------------------------------------------

    @property
    def root(self) -> SplayNode | None:
        return self._root

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: str) -> bool:
        node, _ = self._find(key)
        return node is not None
