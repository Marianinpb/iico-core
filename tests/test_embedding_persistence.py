"""
tests/test_embedding_persistence.py
====================================
Tests de persistencia de embeddings en disco para EmbeddingIndex.

Verifica:
- load_from_disk() carga embeddings .npy sin ONNX
- load_from_disk() con .npy faltantes retorna False
- build_from_chunks() vectoriza chunks sin embedding_path
- build_from_chunks() guarda .npy en disco
- update_chunk() reemplaza embedding existente
- search() funciona tras load_from_disk()
- search() funciona tras build_from_chunks()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iico_core.index.embedding import EmbeddingIndex
from iico_core.types import Chunk


# ============================================================================
# Helpers
# ============================================================================

def _make_embedding(seed: int = 42) -> np.ndarray:
    """Genera un vector de embedding normalizado de dimensión 384."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ============================================================================
# FixedVectorIndex — bypass ONNX para tests
# ============================================================================

class FixedVectorIndex(EmbeddingIndex):
    """EmbeddingIndex con vectorize() que retorna vectores fijos sin ONNX.

    Detecta keywords en el contenido del texto (ignorando el id/tags del
    formato ``<id> <tags> <content>``) para retornar vectores ortogonales:
    - Texto que contiene "topic_a" → vector A (dim=0)
    - Texto que contiene "topic_b" → vector B (dim=1)
    - Texto que contiene "topic_c" → vector C (dim=2)
    - Texto que contiene "topic_d" → vector D (dim=3)
    - Sin keyword reconocida → vector A (default)

    Vectores A/B/C/D son one-hot ortogonales: cosine similarity entre
    vectores distintos = 0.0. El mismo keyword → mismo vector → cosine = 1.0.

    Para que los tests de search() sean deterministas, los chunks en los
    fixtures llevan contenido ``Content of chunk 0``, ``Content of chunk 1``, etc.
    El FixedVectorIndex mapea "chunk 0" → vector B, "chunk 1" → vector C,
    "chunk 2" → vector D, otros → vector A.
    """

    _VEC_DIM = 384

    def _ensure_loaded(self) -> None:
        """Bypass: no carga ONNX ni tokenizador."""
        self._loaded = True

    def vectorize(self, text: str) -> np.ndarray:
        """Retorna vector one-hot según keyword detectado."""
        lower = text.lower()
        # El formato del texto vectorizado es:
        #   "{chunk.id} {' '.join(chunk.tags)} {chunk.content[:512]}"
        # Buscamos keywords solo en el contenido (tras el segundo espacio)
        # Pero para simplificar, buscamos en todo el texto.
        # "chunk 0" aparece tanto en el id como en el contenido → coincide.
        for kw, dim in [
            ("topic_d", 3),
            ("topic_c", 2),
            ("topic_b", 1),
        ]:
            if kw in lower:
                vec = np.zeros(self._VEC_DIM, dtype=np.float32)
                vec[dim] = 1.0
                return vec
        # Default: topic_a → vector A (dim=0)
        vec = np.zeros(self._VEC_DIM, dtype=np.float32)
        vec[0] = 1.0
        return vec


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_chunks_dir(tmp_path: Path) -> Path:
    """Directorio temporal para chunks y sus .npy."""
    chunks_dir = tmp_path / ".chunks"
    chunks_dir.mkdir()
    return chunks_dir


@pytest.fixture
def chunks_with_embeddings(tmp_chunks_dir: Path) -> list[Chunk]:
    """Tres chunks con .npy pre-guardados en disco.

    Cada chunk tiene un keyword único en su contenido para que
    FixedVectorIndex produzca vectores ortogonales:
      chunk_0 → "topic_a" (vector A)
      chunk_1 → "topic_b" (vector B)
      chunk_2 → "topic_c" (vector C)
    """
    topics = ["topic_a", "topic_b", "topic_c"]
    chunks: list[Chunk] = []
    for i, topic in enumerate(topics):
        chunk_id = f"test_note::section_{i}"
        note_dir = tmp_chunks_dir / "test_note"
        note_dir.mkdir(parents=True, exist_ok=True)
        npy_path = note_dir / f"section_{i}.npy"

        vec = _make_embedding(i)
        np.save(str(npy_path), vec)

        chunk = Chunk(
            id=chunk_id,
            parent_note_id="test_note",
            title=f"Section {i}",
            content=f"Content about {topic} with details.",
            tags=["test"],
            priority=5,
            order=i,
            embedding_path=npy_path,
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def chunks_no_embeddings() -> list[Chunk]:
    """Tres chunks sin embedding_path (necesitan vectorización).

    Mismos keywords que chunks_with_embeddings para tests consistentes.
    """
    topics = ["topic_a", "topic_b", "topic_c"]
    chunks: list[Chunk] = []
    for i, topic in enumerate(topics):
        chunk = Chunk(
            id=f"test_note::section_{i}",
            parent_note_id="test_note",
            title=f"Section {i}",
            content=f"Content about {topic} with details.",
            tags=["test"],
            priority=5,
            order=i,
            embedding_path=None,
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def fixed_index() -> FixedVectorIndex:
    """EmbeddingIndex con vectorize() determinista (sin ONNX)."""
    return FixedVectorIndex()


# ============================================================================
# load_from_disk()
# ============================================================================

def test_load_from_disk_loads_all_embeddings(
    chunks_with_embeddings: list[Chunk],
) -> None:
    """Dado chunks con .npy, load_from_disk() carga y devuelve True."""
    index = EmbeddingIndex()
    result = index.load_from_disk(chunks_with_embeddings)

    assert result is True
    assert index.index_size == 3
    assert index._embeddings is not None
    assert index._embeddings.shape == (3, 384)


def test_load_from_disk_with_missing_npy_returns_false(
    chunks_no_embeddings: list[Chunk],
) -> None:
    """Dado chunks sin .npy, load_from_disk() devuelve False, índice vacío."""
    index = EmbeddingIndex()
    result = index.load_from_disk(chunks_no_embeddings)

    assert result is False
    assert index.index_size == 0
    assert index._embeddings is None


def test_load_from_disk_partial_embeddings(
    tmp_chunks_dir: Path,
) -> None:
    """Mezcla de chunks con y sin .npy: carga solo los que tienen embedding."""
    chunks: list[Chunk] = []

    # Chunk 0: con .npy
    note_dir = tmp_chunks_dir / "partial"
    note_dir.mkdir(parents=True, exist_ok=True)
    npy_path_0 = note_dir / "section_0.npy"
    np.save(str(npy_path_0), _make_embedding(0))
    chunks.append(
        Chunk(
            id="partial::section_0",
            parent_note_id="partial",
            title="Section 0",
            content="has embedding topic_a here",
            tags=["test"],
            priority=5,
            order=0,
            embedding_path=npy_path_0,
        )
    )

    # Chunk 1: sin .npy
    chunks.append(
        Chunk(
            id="partial::section_1",
            parent_note_id="partial",
            title="Section 1",
            content="no embedding",
            tags=["test"],
            priority=5,
            order=1,
            embedding_path=None,
        )
    )

    # Chunk 2: con .npy pero el archivo no existe
    missing_path = note_dir / "section_2.npy"
    chunks.append(
        Chunk(
            id="partial::section_2",
            parent_note_id="partial",
            title="Section 2",
            content="missing file",
            tags=["test"],
            priority=5,
            order=2,
            embedding_path=missing_path,  # archivo no creado
        )
    )

    index = EmbeddingIndex()
    result = index.load_from_disk(chunks)

    assert result is True  # al menos uno cargado
    assert index.index_size == 1  # solo section_0
    assert index._notes[0].id == "partial::section_0"


# ============================================================================
# build_from_chunks()
# ============================================================================

def test_build_from_chunks_vectorizes_unembedded(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """Chunks sin embedding_path deben ser vectorizados."""
    fixed_index.build_from_chunks(chunks_no_embeddings)

    assert fixed_index.index_size == 3
    assert fixed_index._embeddings is not None
    assert fixed_index._embeddings.shape == (3, 384)


def test_build_from_chunks_saves_npy_to_disk(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
    tmp_chunks_dir: Path,
) -> None:
    """Tras build_from_chunks con embedding_path, el .npy debe existir en disco."""
    # Asignar embedding_path a cada chunk antes de build
    for i, chunk in enumerate(chunks_no_embeddings):
        note_dir = tmp_chunks_dir / chunk.parent_note_id
        note_dir.mkdir(parents=True, exist_ok=True)
        chunk.embedding_path = note_dir / f"section_{i}.npy"

    fixed_index.build_from_chunks(chunks_no_embeddings)

    for chunk in chunks_no_embeddings:
        assert chunk.embedding_path is not None
        assert chunk.embedding_path.exists(), (
            f"{chunk.embedding_path} debería existir tras build_from_chunks"
        )
        # Verificar que el .npy tiene la dimensión correcta
        loaded = np.load(str(chunk.embedding_path))
        assert loaded.shape == (384,)
        assert loaded.dtype == np.float32


def test_build_from_chunks_force_revectorizes(
    fixed_index: FixedVectorIndex,
    tmp_chunks_dir: Path,
) -> None:
    """force=True debe re-vectorizar incluso si ya existe .npy."""
    # Crear un chunk con .npy pre-existente
    note_dir = tmp_chunks_dir / "force_test"
    note_dir.mkdir(parents=True, exist_ok=True)
    npy_path = note_dir / "section_0.npy"

    # Guardar un vector conocido diferente
    original_vec = _make_embedding(99)
    np.save(str(npy_path), original_vec)

    chunk = Chunk(
        id="force_test::section_0",
        parent_note_id="force_test",
        title="Section 0",
        content="force test content topic_a here",
        tags=["test"],
        priority=5,
        order=0,
        embedding_path=npy_path,
    )

    fixed_index.build_from_chunks([chunk], force=True)

    # El vector guardado ahora debe ser diferente (generado por FixedVectorIndex)
    loaded = np.load(str(npy_path))
    # FixedVectorIndex retorna vector one-hot dim=0 para "topic_a"
    assert loaded[0] == 1.0
    assert loaded[1] == 0.0

    # Sin force, el vector original debe preservarse
    np.save(str(npy_path), original_vec)
    fixed_index.build_from_chunks([chunk], force=False)
    loaded_no_force = np.load(str(npy_path))
    np.testing.assert_array_equal(loaded_no_force, original_vec)


# ============================================================================
# update_chunk()
# ============================================================================

def test_update_chunk_replaces_existing_embedding(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """update_chunk() debe reemplazar la fila del embedding existente."""
    fixed_index.build_from_chunks(chunks_no_embeddings)

    # Guardar el embedding original del chunk 1 (vector B, dim=1)
    original_emb = fixed_index._embeddings[1].copy() if fixed_index._embeddings is not None else None
    assert original_emb is not None

def test_update_chunk_replaces_existing_embedding(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """update_chunk() debe reemplazar la fila del embedding existente."""
    fixed_index.build_from_chunks(chunks_no_embeddings)

    # section_1 tiene "topic_b" → vector B (dim=1)
    original_emb = fixed_index._embeddings[1].copy() if fixed_index._embeddings is not None else None
    assert original_emb is not None

    # Modificar section_1: cambiar contenido a "topic_d" → vector D (dim=3)
    modified_chunk = Chunk(
        id=chunks_no_embeddings[1].id,  # mismo id → reemplaza la fila
        parent_note_id=chunks_no_embeddings[1].parent_note_id,
        title=chunks_no_embeddings[1].title,
        content="updated content about topic_d here",
        tags=["updated"],
        priority=9,
        order=chunks_no_embeddings[1].order,
        embedding_path=None,
    )

    fixed_index.update_chunk(modified_chunk)

    # Verificar que el embedding cambió (topic_d → vector D, dim=3 ≠ dim=1)
    assert fixed_index._embeddings is not None
    new_emb = fixed_index._embeddings[1]
    assert not np.array_equal(new_emb, original_emb)
    # El chunk en _notes debe ser el modificado
    assert fixed_index._notes[1].content == "updated content about topic_d here"
    assert fixed_index._notes[1].priority == 9


def test_update_chunk_appends_new_chunk(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """update_chunk() con chunk nuevo debe agregarlo al índice."""
    fixed_index.build_from_chunks(chunks_no_embeddings)
    assert fixed_index.index_size == 3

    new_chunk = Chunk(
        id="test_note::chunk_new",
        parent_note_id="test_note",
        title="New Chunk",
        content="Brand new content",
        tags=["new"],
        priority=7,
        order=99,
        embedding_path=None,
    )

    fixed_index.update_chunk(new_chunk)
    assert fixed_index.index_size == 4
    assert fixed_index._notes[-1].id == "test_note::chunk_new"


# ============================================================================
# search() después de persistencia
# ============================================================================

def test_search_works_after_load_from_disk(
    chunks_with_embeddings: list[Chunk],
    fixed_index: FixedVectorIndex,
) -> None:
    """search() debe funcionar tras load_from_disk() sin cargar ONNX."""
    # load_from_disk no usa ONNX, así que usamos EmbeddingIndex normal
    index = EmbeddingIndex()
    index.load_from_disk(chunks_with_embeddings)

    # search() necesita vectorize para el query, usamos fixed_index
    # (que tiene _ensure_loaded bypass) para poder vectorizar el query.
    fixed_index.load_from_disk(chunks_with_embeddings)

    # Buscar por keyword "topic_a" → debe matchear section_0
    results = fixed_index.search("topic_a information", threshold=0.0, top_k=3)
    assert len(results) >= 1
    best_chunk, best_score = results[0]
    assert best_chunk.id == "test_note::section_0"


def test_search_works_after_build_from_chunks(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """search() debe funcionar tras build_from_chunks()."""
    fixed_index.build_from_chunks(chunks_no_embeddings)

    # Buscar por keyword "topic_a" → debe matchear section_0
    results = fixed_index.search("topic_a information", threshold=0.0, top_k=3)
    assert len(results) >= 1
    best_chunk, best_score = results[0]
    assert best_chunk.id == "test_note::section_0"


def test_search_threshold_filters_results(
    fixed_index: FixedVectorIndex,
    chunks_no_embeddings: list[Chunk],
) -> None:
    """Umbral alto debe filtrar resultados con baja similitud."""
    fixed_index.build_from_chunks(chunks_no_embeddings)

    # Con threshold=0.999, solo resultados casi idénticos pasan
    # "topic_a" query → vector A. Solo section_0 (topic_a) → cosine=1.0.
    # section_1 (topic_b) → vector B → cosine=0.0 (ortogonal).
    results = fixed_index.search("topic_a", threshold=0.999, top_k=5)
    assert len(results) == 1
    assert results[0][0].id == "test_note::section_0"


# ============================================================================
# Verificación de regresiones: build_index deprecado sigue funcionando
# ============================================================================

def test_build_index_still_works_with_warning(
    fixed_index: FixedVectorIndex,
) -> None:
    """build_index() deprecado debe seguir funcionando con PassiveNote."""
    from iico_core.memory.passive import PassiveNote

    notes = [
        PassiveNote(id="a", tags=["x"], priority=5, content="content a"),
        PassiveNote(id="b", tags=["y"], priority=3, content="content b"),
    ]

    with pytest.warns(DeprecationWarning, match="build_index"):
        fixed_index.build_index(notes)

    assert fixed_index.index_size == 2
    # search() debe funcionar tras build_index deprecado
    results = fixed_index.search("content a", threshold=0.0, top_k=2)
    assert len(results) >= 1
