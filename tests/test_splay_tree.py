"""
tests/test_splay_tree.py
=========================
Tests unitarios del Splay Tree y SplayCacheMetrics.
"""

import pytest
from iico_core.index.splay_tree import SplayCacheMetrics, SplayNode, SplayTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tree():
    return SplayTree(max_nodes=10)


@pytest.fixture
def small_tree():
    """Árbol con 5 nodos pre-insertados."""
    t = SplayTree(max_nodes=10)
    for k in ["e", "b", "d", "a", "c"]:
        t.insert(k, f"val_{k}")
    return t


# ---------------------------------------------------------------------------
# Insert y search básicos
# ---------------------------------------------------------------------------

def test_insert_single(tree):
    tree.insert("key1", "value1")
    assert tree.size == 1
    assert tree.root is not None
    assert tree.root.key == "key1"


def test_insert_multiple(tree):
    for k in ["c", "a", "b"]:
        tree.insert(k, k)
    assert tree.size == 3


def test_search_existing(tree):
    tree.insert("alpha", "data_alpha")
    tree.insert("beta", "data_beta")
    result = tree.search("alpha")
    assert result is not None
    assert result.key == "alpha"
    assert result.value == "data_alpha"


def test_search_missing(tree):
    tree.insert("a", "v")
    result = tree.search("z")
    assert result is None


def test_search_splayeado_a_raiz(small_tree):
    """Después de search, el nodo encontrado debe ser la raíz."""
    small_tree.search("b")
    assert small_tree.root is not None
    assert small_tree.root.key == "b"


def test_insert_updates_existing(tree):
    """Insertar una key ya existente actualiza el valor sin duplicar."""
    tree.insert("k", "old")
    tree.insert("k", "new")
    assert tree.size == 1
    result = tree.search("k")
    assert result.value == "new"


# ---------------------------------------------------------------------------
# Operación de Splay (corrección de rotaciones)
# ---------------------------------------------------------------------------

def test_splay_root_property(tree):
    """Tras insertar varios nodos, el último insertado es la raíz."""
    tree.insert("m", 1)
    tree.insert("a", 2)
    tree.insert("z", 3)
    # El último insertado se splayea a la raíz
    assert tree.root.key == "z"


def test_parent_pointers_consistent(tree):
    """Verifica que los parent pointers son correctos tras splaying."""
    for k in ["d", "b", "f", "a", "c", "e", "g"]:
        tree.insert(k, k)

    def check_parents(node, parent=None):
        if node is None:
            return
        assert node.parent is parent, f"Parent incorrecto en {node.key}"
        check_parents(node.left, node)
        check_parents(node.right, node)

    check_parents(tree.root)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_existing(small_tree):
    initial_size = small_tree.size
    deleted = small_tree.delete("b")
    assert deleted is True
    assert small_tree.size == initial_size - 1
    assert small_tree.search("b") is None


def test_delete_missing(small_tree):
    deleted = small_tree.delete("z")
    assert deleted is False


def test_delete_root(tree):
    tree.insert("only", "v")
    tree.delete("only")
    assert tree.size == 0
    assert tree.root is None


def test_delete_all(small_tree):
    keys = [n.key for n in small_tree.peek_top(small_tree.size + 5)]
    # Eliminar todos los nodos
    for k in ["a", "b", "c", "d", "e"]:
        small_tree.delete(k)
    assert small_tree.size == 0


# ---------------------------------------------------------------------------
# Evicción por capacidad
# ---------------------------------------------------------------------------

def test_eviction_respects_max_nodes():
    """El árbol no debe superar max_nodes nodos."""
    t = SplayTree(max_nodes=5)
    for i in range(10):
        t.insert(f"key{i}", i)
    assert t.size <= 5


def test_contains_operator(tree):
    tree.insert("present", 1)
    assert "present" in tree
    assert "absent" not in tree


# ---------------------------------------------------------------------------
# peek_top
# ---------------------------------------------------------------------------

def test_peek_top_returns_up_to_n(small_tree):
    top = small_tree.peek_top(3)
    assert len(top) <= 3


def test_peek_keys_top(small_tree):
    keys = small_tree.peek_keys_top(3)
    assert isinstance(keys, set)
    assert len(keys) <= 3


# ---------------------------------------------------------------------------
# SplayCacheMetrics
# ---------------------------------------------------------------------------

def test_metrics_hit_rate_empty():
    m = SplayCacheMetrics()
    assert m.hit_rate == 0.0
    assert m.avg_depth == 0.0


def test_metrics_records_hits_and_misses():
    m = SplayCacheMetrics()
    m.record_access(depth=0, hit=True)
    m.record_access(depth=1, hit=True)
    m.record_access(depth=3, hit=False)
    assert m.hits == 2
    assert m.misses == 1
    assert m.total_accesses == 3
    assert abs(m.hit_rate - 2 / 3) < 1e-6
    assert abs(m.avg_depth - (0 + 1 + 3) / 3) < 1e-6


def test_metrics_integrated_with_tree():
    """Las métricas se registran automáticamente con search."""
    m = SplayCacheMetrics()
    t = SplayTree(max_nodes=10, metrics=m)
    t.insert("x", 1)
    t.search("x")   # hit
    t.search("y")   # miss
    assert m.hits == 1
    assert m.misses == 1


def test_metrics_summary_keys():
    m = SplayCacheMetrics()
    summary = m.summary()
    assert "hits" in summary
    assert "misses" in summary
    assert "hit_rate" in summary
    assert "avg_depth" in summary


def test_metrics_reset():
    m = SplayCacheMetrics()
    m.record_access(1, hit=True)
    m.reset()
    assert m.hits == 0
    assert m.total_accesses == 0
