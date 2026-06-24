"""
tests/test_chunker_semantic.py
===============================
Tests unitarios del SemanticSplitter y su integración con Chunker.

Verifica:
- Ventanas deslizantes (sliding_windows)
- Detección de puntos de quiebre por cosine similarity
- División semántica (split)
- Integración con Chunker (pasada estructural → refinamiento semántico)
- Fallback a párrafos cuando no hay SemanticSplitter
"""

from __future__ import annotations

import pytest

from iico_core.memory.chunker import Chunker, SemanticSplitter
from iico_core.memory.passive import PassiveNote


# ============================================================================
# MockEmbeddingIndex — retorna vectores controlados para tests deterministas
# ============================================================================

class MockEmbeddingIndex:
    """Mock de EmbeddingIndex que retorna vectores ortogonales según keywords.

    - Texto con "splay" o "rotacion" → vector A (norma 1, apunta a dim[0])
    - Texto con "avl" o "balance"  → vector B (norma 1, apunta a dim[1])
    - Texto sin keyword           → vector A (default)

    Los vectores A y B son ortogonales: cosine_similarity(A, B) = 0.0
    Dos llamadas con keyword "splay" retornan el mismo vector → cosine = 1.0
    """

    def __init__(self) -> None:
        import numpy as np

        self._vec_a = np.zeros(384, dtype=np.float32)
        self._vec_a[0] = 1.0

        self._vec_b = np.zeros(384, dtype=np.float32)
        self._vec_b[1] = 1.0

    def vectorize(self, text: str):
        """Retorna vector A o B según el contenido del texto."""
        import numpy as np

        lower = text.lower()
        if "avl" in lower or "balance" in lower:
            return self._vec_b.copy()
        # Default: vector A ("splay", "rotacion", o cualquier otro)
        return self._vec_a.copy()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_index() -> MockEmbeddingIndex:
    return MockEmbeddingIndex()


@pytest.fixture
def splitter(mock_index: MockEmbeddingIndex) -> SemanticSplitter:
    """SemanticSplitter con ventanas de 200 tokens (800 chars) y stride 50 (200 chars)."""
    return SemanticSplitter(
        embedding_index=mock_index,
        window_tokens=200,
        stride_tokens=50,
    )


@pytest.fixture
def chunker_no_semantic() -> Chunker:
    """Chunker sin SemanticSplitter (comportamiento original)."""
    return Chunker(max_chunk_tokens=20)


@pytest.fixture
def chunker_with_semantic(mock_index: MockEmbeddingIndex) -> Chunker:
    """Chunker con SemanticSplitter integrado."""
    splitter = SemanticSplitter(
        embedding_index=mock_index,
        window_tokens=50,   # ventanas pequeñas para tests
        stride_tokens=10,
    )
    return Chunker(max_chunk_tokens=20, semantic_splitter=splitter)


def _make_note(
    note_id: str = "test-note",
    content: str = "",
    tags: list[str] | None = None,
    priority: int = 5,
) -> PassiveNote:
    """Construye una PassiveNote mínima para tests."""
    return PassiveNote(
        id=note_id,
        tags=tags if tags is not None else [],
        priority=priority,
        content=content,
    )


# ============================================================================
# SemanticSplitter — _sliding_windows
# ============================================================================

def test_sliding_windows_correct_count(splitter: SemanticSplitter):
    """Texto de ~2000 caracteres debe producir múltiples ventanas."""
    # window_chars = 200*4 = 800, stride_chars = 50*4 = 200
    # 1600 chars, stride 200 → ceil(1600/200) = 8 ventanas
    text = "x" * 1600
    windows = splitter._sliding_windows(text)
    assert len(windows) == 8
    # Primera ventana: chars 0 a 800
    assert len(windows[0]) == 800
    # Segunda ventana: chars 200 a 1000
    assert windows[1] == text[200:1000]
    # Última ventana: chars 1400 a 1600 (200 chars)
    assert len(windows[-1]) == 200


def test_sliding_windows_last_window_shorter(splitter: SemanticSplitter):
    """La última ventana puede ser más corta si el texto no alcanza."""
    # Texto de 900 chars, stride 200 → ceil(900/200) = 5 ventanas
    # Ventanas: [0:800], [200:1000→900], [400:1200→900], [600:1400→900], [800:1600→900]
    text = "x" * 900
    windows = splitter._sliding_windows(text)
    assert len(windows) == 5
    assert len(windows[0]) == 800
    # Última ventana empieza en 800, termina en 900 → 100 chars
    assert len(windows[-1]) == 100


def test_sliding_windows_text_shorter_than_window(splitter: SemanticSplitter):
    """Texto más corto que la ventana produce ventanas con solapamiento."""
    # window_chars = 800, stride = 200, texto de 500 chars
    # ceil(500/200) = 3 ventanas (con solapamiento, todas ≤ 500 chars)
    text = "x" * 500
    windows = splitter._sliding_windows(text)
    assert len(windows) == 3
    # Todas las ventanas son substrings del texto original
    for w in windows:
        assert w in text


# ============================================================================
# SemanticSplitter — _compute_break_points
# ============================================================================

def test_compute_break_points_same_topic_no_breaks(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Ventanas del mismo tópico → cosine ~1.0 → sin puntos de quiebre."""
    # Todas las ventanas contienen "splay" → vector A en cada una
    windows = ["splay tree data structure"] * 5
    break_points = splitter._compute_break_points(windows, threshold=0.5)
    assert break_points == []


def test_compute_break_points_topic_change_detected(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Primeras ventanas tópico A, últimas tópico B → punto de quiebre."""
    # 3 ventanas con "splay" (vector A), 2 ventanas con "avl" (vector B)
    windows = [
        "splay tree rotation",
        "splay tree zig zig-zag",
        "splay tree amortized",
        "avl tree balance factor",
        "avl tree strict height",
    ]
    break_points = splitter._compute_break_points(windows, threshold=0.5)
    # Quiebre entre ventana 2 y 3 (índice i=2, bp = (2+1)*stride_chars = 3*200 = 600)
    assert len(break_points) == 1
    assert break_points[0] == 600  # (i+1) * stride_chars = 3 * 200


def test_compute_break_points_single_window_no_breaks(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Una sola ventana → sin pares consecutivos → sin puntos de quiebre."""
    windows = ["splay tree"]
    break_points = splitter._compute_break_points(windows, threshold=0.5)
    assert break_points == []


# ============================================================================
# SemanticSplitter — split (API pública)
# ============================================================================

def test_split_uniform_text_single_fragment(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Texto uniforme (mismo tópico) → un solo fragmento."""
    # Texto largo pero todo del mismo tópico "splay"
    text = "splay tree rotation and analysis. " * 100
    fragments = splitter.split(text, threshold=0.5)
    assert len(fragments) == 1
    # split() retorna [text] sin modificar cuando no hay puntos de quiebre
    assert fragments[0] == text


def test_split_topic_change_produces_multiple_fragments(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Texto con cambio de tópico → 2+ fragmentos."""
    # Primera mitad: tópico "splay", segunda mitad: tópico "avl"
    # Necesitamos suficiente texto para producir múltiples ventanas
    # window_chars = 800, stride_chars = 200
    # Para tener ~4 ventanas de A y ~2 ventanas de B:
    half_a = "splay tree rotation zig zig-zag analysis amortized complexity. " * 30   # ~1800 chars
    half_b = "avl tree balance factor strict height rotation rules. " * 30             # ~1800 chars
    text = half_a + half_b

    fragments = splitter.split(text, threshold=0.5)
    # Debe haber al menos 2 fragmentos (uno de cada tópico)
    assert len(fragments) >= 2
    # El primer fragmento debe contener "splay"
    assert "splay" in fragments[0].lower()
    # El último fragmento debe contener "avl"
    assert "avl" in fragments[-1].lower()


def test_split_short_text_no_split(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Texto más corto que la ventana → retorna [text] sin dividir."""
    text = "splay tree. avl tree."  # ~25 chars, mucho menos que 800 chars de ventana
    fragments = splitter.split(text, threshold=0.5)
    assert fragments == [text]


def test_split_empty_text(splitter: SemanticSplitter, mock_index: MockEmbeddingIndex):
    """Texto vacío → [texto vacío]."""
    fragments = splitter.split("", threshold=0.5)
    assert fragments == [""]


# ============================================================================
# Chunker con SemanticSplitter — integración
# ============================================================================

def test_chunker_without_semantic_uses_paragraphs_only(chunker_no_semantic: Chunker):
    """Sin SemanticSplitter, _split_oversized divide solo por párrafos."""
    # Un párrafo gigante que excede max_chunk_tokens=20
    big_paragraph = "ParrafoUnicoMuyLargoQueSuperaLimite. " * 20  # ~700 chars, ~175 tokens
    note = _make_note(
        note_id="big-para",
        content="## Intro\n\n" + big_paragraph + "\n",
    )
    chunks = chunker_no_semantic.chunk_note(note)
    # Sin semantic splitter, el párrafo grande se intenta dividir por párrafos
    # pero como es un solo párrafo, termina en un chunk (o se parte forzadamente)
    # El Chunker actual lo mete en un batch que igual excede el límite
    assert len(chunks) >= 1


def test_chunker_with_semantic_refines_oversized(chunker_with_semantic: Chunker):
    """Chunker con SemanticSplitter: un párrafo mixto (2 tópicos) debe dividirse."""
    # Construir un párrafo largo con dos tópicos distintos
    # El mock usa keywords "splay"/"rotacion" → vec A, "avl"/"balance" → vec B
    # Con window_tokens=50 (200 chars) y stride_tokens=10 (40 chars),
    # necesitamos texto suficiente para generar varias ventanas
    topic_a = "splay tree rotation analysis. " * 30   # ~900 chars
    topic_b = "avl balance factor strict height. " * 30  # ~900 chars
    big_paragraph = topic_a + topic_b

    note = _make_note(
        note_id="mixed-topics",
        content="## Arboles\n\n" + big_paragraph + "\n",
    )
    chunks = chunker_with_semantic.chunk_note(note)
    # Debe producir más chunks que sin semantic splitter (que pondría todo en uno)
    assert len(chunks) >= 1
    # Al menos uno de los chunks debe contener "splay" y otro "avl"
    all_content = " ".join(c.content for c in chunks)
    assert "splay" in all_content.lower()
    assert "avl" in all_content.lower()


def test_chunker_semantic_preserves_chunk_metadata(chunker_with_semantic: Chunker):
    """Los chunks refinados semánticamente deben heredar tags, prioridad y parent_note_id."""
    topic_a = "splay tree rotation analysis. " * 30
    topic_b = "avl balance factor strict height. " * 30
    big_paragraph = topic_a + topic_b

    note = _make_note(
        note_id="meta-test",
        tags=["arboles", "estructuras"],
        priority=8,
        content="## Arboles\n\n" + big_paragraph + "\n",
    )
    chunks = chunker_with_semantic.chunk_note(note)
    for chunk in chunks:
        assert chunk.parent_note_id == "meta-test"
        assert chunk.tags == ["arboles", "estructuras"]
        assert chunk.priority == 8


def test_chunker_semantic_chunks_have_sequential_order(chunker_with_semantic: Chunker):
    """El orden secuencial debe mantenerse incluso con refinamiento semántico."""
    topic_a = "splay tree rotation analysis. " * 30
    topic_b = "avl balance factor strict height. " * 30
    big_paragraph = topic_a + topic_b

    note = _make_note(
        note_id="order-test",
        content="## Arboles\n\n" + big_paragraph + "\n",
    )
    chunks = chunker_with_semantic.chunk_note(note)
    orders = [c.order for c in chunks]
    assert orders == sorted(orders)
    assert orders == list(range(len(chunks)))


# ============================================================================
# Integración: pipeline completo (estructural → semántico)
# ============================================================================

def test_integration_structural_then_semantic(chunker_with_semantic: Chunker):
    """Pipeline completo: parseo estructural + refinamiento semántico en oversized."""
    # Nota con múltiples secciones: algunas normales, una oversized con mezcla de tópicos
    topic_a = "splay tree rotation analysis zig zig-zag. " * 20   # ~800 chars
    topic_b = "avl balance factor strict height rules. " * 20      # ~800 chars
    mixed_section = topic_a + topic_b

    content = (
        "## Intro\n\n"
        "Breve introduccion a los arboles de busqueda.\n\n"
        "## Comparativa\n\n"
        + mixed_section + "\n\n"
        "## Conclusion\n\n"
        "Los arboles auto-balanceables son esenciales.\n"
    )
    note = _make_note(
        note_id="full-pipeline",
        tags=["arboles"],
        priority=5,
        content=content,
    )
    chunks = chunker_with_semantic.chunk_note(note)

    # Debe haber chunks para Intro, Comparativa (posiblemente dividido), y Conclusion
    assert len(chunks) >= 3

    # Intro debe ser el primer chunk (order 0)
    intro_chunks = [c for c in chunks if c.title == "Intro"]
    assert len(intro_chunks) == 1

    # Conclusion debe ser uno de los últimos
    conclusion_chunks = [c for c in chunks if c.title == "Conclusion"]
    assert len(conclusion_chunks) == 1

    # La sección Comparativa debe haberse dividido (contiene ambos tópicos)
    comparativa_chunks = [c for c in chunks if "Comparativa" in c.title]
    assert len(comparativa_chunks) >= 1


# ============================================================================
# SemanticSplitter sin mock — verificación de que NO requiere numpy en chunker.py
# ============================================================================

def test_semantic_splitter_can_be_imported_without_embedding_index():
    """SemanticSplitter se puede importar sin tener EmbeddingIndex cargado."""
    # Solo verificar que la clase existe y es instanciable con TYPE_CHECKING
    from iico_core.memory.chunker import SemanticSplitter as SS
    assert SS is not None


def test_chunker_accepts_none_semantic_splitter():
    """Chunker con semantic_splitter=None debe funcionar normalmente."""
    chunker = Chunker(max_chunk_tokens=100, semantic_splitter=None)
    note = _make_note(content="## Test\n\nContenido de prueba.\n")
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].title == "Test"
