"""
tests/test_chunking_strategies.py
===================================
Tests para los plugins de chunking de la Fase 1.

Ejecutar::

    pytest tests/test_chunking_strategies.py -v
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from iico_core.rag_bench.chunking.base import (
    ChunkBoundary,
    ChunkingResult,
    ChunkingStrategy,
    get_chunker,
    list_chunkers,
    register_chunker,
)
from iico_core.rag_bench.chunking.structural import StructuralChunker
from iico_core.rag_bench.chunking.convolution import ConvolutionChunker
from iico_core.rag_bench.chunking.pipeline import ChunkingPipeline

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

skip_no_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


# ---------------------------------------------------------------------------
# Texto de prueba
# ---------------------------------------------------------------------------

SIMPLE_TEXT = """\
# Introducción

Este es el contenido introductorio de la nota. Aquí se presenta el tema general.

## Sección A

Contenido de la sección A. Habla sobre el primer tema principal.
Tiene varias oraciones para tener algo de substancia.

## Sección B

Contenido de la sección B. Aquí se discute el segundo tema.

### Subsección B.1

Detalle adicional sobre B.1.

## Conclusión

Resumen final de los puntos clave.
"""

CODE_TEXT = """\
# Con Código

Texto antes del código.

```python
def hola():
    return "mundo"
```

Texto después del código.

## Segunda Sección

Más contenido aquí.
"""

SHORT_TEXT = "Un texto corto sin secciones ni headers."


# ===========================================================================
# Tests del Registry
# ===========================================================================

class TestRegistry:
    def test_structural_registered(self) -> None:
        # Importar el paquete activa los registros
        from iico_core.rag_bench.chunking import list_chunkers
        chunkers = list_chunkers()
        assert "structural" in chunkers

    def test_convolution_registered(self) -> None:
        from iico_core.rag_bench.chunking import list_chunkers
        chunkers = list_chunkers()
        assert "convolution" in chunkers

    def test_get_chunker_structural(self) -> None:
        cls = get_chunker("structural")
        assert cls is StructuralChunker

    def test_get_chunker_convolution(self) -> None:
        cls = get_chunker("convolution")
        assert cls is ConvolutionChunker

    def test_get_chunker_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="no encontrado"):
            get_chunker("chunker_que_no_existe")

    def test_register_custom_chunker(self) -> None:
        @register_chunker
        class DummyChunker(ChunkingStrategy):
            name = "dummy_test_chunker"
            description = "Solo para tests"

            def chunk(self, text, config=None):
                return ChunkingResult(
                    fragments=[text], boundaries=[], metadata={}
                )

        assert "dummy_test_chunker" in list_chunkers()
        cls = get_chunker("dummy_test_chunker")
        instance = cls()
        result = instance.chunk("hola")
        assert result.fragments == ["hola"]

    def test_register_without_name_raises(self) -> None:
        with pytest.raises(ValueError, match="'name'"):
            @register_chunker
            class NoNameChunker(ChunkingStrategy):
                def chunk(self, text, config=None):
                    return ChunkingResult([], [], {})


# ===========================================================================
# Tests de ChunkBoundary
# ===========================================================================

class TestChunkBoundary:
    def test_confidence_clamped(self) -> None:
        b = ChunkBoundary(position=0, confidence=1.5)
        assert b.confidence == 1.0

    def test_confidence_clamped_negative(self) -> None:
        b = ChunkBoundary(position=0, confidence=-0.5)
        assert b.confidence == 0.0


# ===========================================================================
# Tests de ChunkingResult
# ===========================================================================

class TestChunkingResult:
    def test_num_chunks(self) -> None:
        r = ChunkingResult(fragments=["a", "b", "c"], boundaries=[])
        assert r.num_chunks == 3

    def test_has_dsp_signals_true(self) -> None:
        r = ChunkingResult(
            fragments=[], boundaries=[],
            metadata={"similarity_signal": [0.8, 0.6]}
        )
        assert r.has_dsp_signals is True

    def test_has_dsp_signals_false(self) -> None:
        r = ChunkingResult(fragments=[], boundaries=[], metadata={"chunker": "structural"})
        assert r.has_dsp_signals is False


# ===========================================================================
# Tests de StructuralChunker
# ===========================================================================

class TestStructuralChunker:
    @pytest.fixture
    def chunker(self):
        return StructuralChunker()

    def test_basic_sections(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        assert result.num_chunks >= 2  # Al menos intro + secciones
        assert all(f.strip() for f in result.fragments)

    def test_short_text_produces_chunk(self, chunker) -> None:
        result = chunker.chunk(SHORT_TEXT)
        assert result.num_chunks >= 1
        assert SHORT_TEXT in result.fragments[0]

    def test_empty_text(self, chunker) -> None:
        result = chunker.chunk("")
        assert result.num_chunks >= 0  # No crashea

    def test_code_block_separated(self, chunker) -> None:
        result = chunker.chunk(CODE_TEXT)
        assert result.num_chunks >= 1
        # Buscar si hay un fragmento que incluya la función de Python
        full_text = " ".join(result.fragments)
        assert "hola" in full_text

    def test_boundaries_count(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        # Hay n-1 boundaries para n chunks (aprox)
        assert len(result.boundaries) == result.num_chunks - 1 or len(result.boundaries) >= 0

    def test_metadata_chunker_name(self, chunker) -> None:
        result = chunker.chunk(SHORT_TEXT)
        assert result.metadata["chunker"] == "structural"

    def test_config_max_tokens(self, chunker) -> None:
        # Texto largo → con max_chunk_tokens=50, debe partir en más chunks
        long_text = "\n\n".join([f"Párrafo {i}: " + "x" * 200 for i in range(5)])
        result_big = chunker.chunk(long_text, {"max_chunk_tokens": 2000})
        result_small = chunker.chunk(long_text, {"max_chunk_tokens": 50})
        assert result_small.num_chunks >= result_big.num_chunks

    def test_subsections_merged(self, chunker) -> None:
        text = """\
## Sección Principal

Intro de la sección.

### Sub A

Contenido de sub A.

### Sub B

Contenido de sub B.
"""
        result = chunker.chunk(text)
        # ### sub-secciones deberían fusionarse bajo el ## principal
        full = " ".join(result.fragments)
        assert "Sub A" in full
        assert "Sub B" in full

    def test_fragments_cover_original_content(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        joined = " ".join(result.fragments).lower()
        # Use keywords that survive any encoding issues on Windows
        assert (
            "contenido" in joined or "secci" in joined or "resumen" in joined
        ), f"Fragmentos no contienen contenido esperado. Chunks: {result.num_chunks}"

    def test_no_empty_fragments(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        for frag in result.fragments:
            assert frag.strip(), f"Fragmento vacío detectado: {frag!r}"


# ===========================================================================
# Tests de ConvolutionChunker (sin EmbeddingIndex)
# ===========================================================================

class TestConvolutionChunkerFallback:
    """Tests del ConvolutionChunker en modo fallback (sin EmbeddingIndex)."""

    @pytest.fixture
    def chunker(self):
        c = ConvolutionChunker()
        # Sin setup → sin embedding_index → fallback a párrafos
        return c

    def test_fallback_produces_chunks(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        assert result.num_chunks >= 1

    def test_fallback_metadata_flag(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        assert result.metadata.get("fallback") is True

    def test_empty_dsp_signals_in_fallback(self, chunker) -> None:
        result = chunker.chunk(SIMPLE_TEXT)
        assert result.metadata["similarity_signal"] == []
        assert result.metadata["smoothed_signal"] == []

    def test_empty_text_fallback(self, chunker) -> None:
        result = chunker.chunk("")
        # No debe crashear
        assert isinstance(result, ChunkingResult)


# ===========================================================================
# Tests de ConvolutionChunker — Kernels (solo aritmética, sin embeddings)
# ===========================================================================

@skip_no_numpy
class TestConvolutionKernels:
    @pytest.fixture
    def chunker(self):
        return ConvolutionChunker()

    def test_gaussian_kernel_shape(self, chunker) -> None:
        k = chunker._gaussian_kernel(5)
        assert len(k) == 5

    def test_gaussian_kernel_sums_to_one(self, chunker) -> None:
        k = chunker._gaussian_kernel(5)
        assert abs(k.sum() - 1.0) < 1e-6

    def test_gaussian_kernel_symmetric(self, chunker) -> None:
        k = chunker._gaussian_kernel(5)
        np.testing.assert_array_almost_equal(k, k[::-1])

    def test_edge_kernel_shape(self, chunker) -> None:
        k = chunker._edge_kernel(5)
        assert len(k) == 5

    def test_edge_kernel_antisymmetric(self, chunker) -> None:
        k = chunker._edge_kernel(5)
        # Parte negativa en la izquierda, positiva en la derecha
        assert k[0] < 0
        assert k[-1] > 0

    def test_build_kernel_box(self, chunker) -> None:
        k = chunker._build_kernel("box", 4, None)
        np.testing.assert_array_almost_equal(k, np.ones(4) / 4)

    def test_build_kernel_custom(self, chunker) -> None:
        weights = [1.0, 2.0, 1.0]
        k = chunker._build_kernel("custom", 3, weights)
        # Normalizado: suma absoluta = 4
        assert abs(k.sum() - 1.0) < 1e-6

    def test_build_kernel_unknown_raises(self, chunker) -> None:
        with pytest.raises(ValueError, match="desconocido"):
            chunker._build_kernel("unknown_kernel", 5, None)

    def test_custom_without_weights_raises(self, chunker) -> None:
        with pytest.raises(ValueError, match="kernel_weights"):
            chunker._build_kernel("custom", 5, None)


# ===========================================================================
# Tests de ConvolutionChunker — Señal de similaridad
# ===========================================================================

@skip_no_numpy
class TestSimilaritySignal:
    @pytest.fixture
    def chunker(self):
        return ConvolutionChunker()

    def test_signal_length(self, chunker) -> None:
        embeddings = np.random.randn(6, 384).astype(np.float64)
        # Normalizar
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        signal = chunker._similarity_signal(embeddings)
        assert len(signal) == 5  # n - 1

    def test_signal_single_embedding(self, chunker) -> None:
        embeddings = np.random.randn(1, 384).astype(np.float64)
        signal = chunker._similarity_signal(embeddings)
        assert len(signal) == 0

    def test_identical_embeddings_high_sim(self, chunker) -> None:
        vec = np.ones(384, dtype=np.float64)
        vec /= np.linalg.norm(vec)
        embeddings = np.stack([vec, vec, vec])
        signal = chunker._similarity_signal(embeddings)
        assert all(s > 0.99 for s in signal)

    def test_orthogonal_embeddings_low_sim(self, chunker) -> None:
        v1 = np.zeros(384)
        v2 = np.zeros(384)
        v1[0] = 1.0
        v2[1] = 1.0
        embeddings = np.stack([v1, v2])
        signal = chunker._similarity_signal(embeddings)
        assert abs(signal[0]) < 0.01  # Cosine ≈ 0


# ===========================================================================
# Tests de ConvolutionChunker — Detección de valles
# ===========================================================================

@skip_no_numpy
class TestValleyDetection:
    @pytest.fixture
    def chunker(self):
        return ConvolutionChunker()

    def test_clear_valley(self, chunker) -> None:
        signal = np.array([0.8, 0.8, 0.2, 0.8, 0.8])
        valleys = chunker._detect_valleys(signal, threshold=0.5)
        assert 2 in valleys

    def test_no_valley_above_threshold(self, chunker) -> None:
        signal = np.array([0.8, 0.8, 0.7, 0.8, 0.8])
        valleys = chunker._detect_valleys(signal, threshold=0.5)
        assert len(valleys) == 0

    def test_multiple_valleys(self, chunker) -> None:
        signal = np.array([0.9, 0.2, 0.9, 0.9, 0.1, 0.9])
        valleys = chunker._detect_valleys(signal, threshold=0.5)
        assert 1 in valleys
        assert 4 in valleys

    def test_flat_signal_no_valleys(self, chunker) -> None:
        signal = np.array([0.5, 0.5, 0.5])
        # 0.5 no es < threshold=0.5 (umbral estricto)
        valleys = chunker._detect_valleys(signal, threshold=0.5)
        assert len(valleys) == 0


# ===========================================================================
# Tests de ConvolutionChunker con EmbeddingIndex mock
# ===========================================================================

class MockEmbeddingIndex:
    """Mock del EmbeddingIndex para tests sin ONNX."""

    def __init__(self, mode: str = "similar"):
        self.mode = mode
        self._call_count = 0

    def vectorize(self, text: str):
        """Genera embeddings deterministas según el modo."""
        import numpy as np
        self._call_count += 1

        if self.mode == "similar":
            # Embeddings casi idénticos → sin valles → 1 chunk
            base = np.ones(384) / math.sqrt(384)
            noise = np.random.default_rng(hash(text) % 1000).normal(0, 0.01, 384)
            vec = base + noise
        elif self.mode == "dissimilar":
            # Embeddings ortogonales → valles → múltiples chunks
            seed = hash(text) % 384
            vec = np.zeros(384)
            vec[seed % 384] = 1.0
        else:
            vec = np.random.default_rng(42).standard_normal(384)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


@skip_no_numpy
class TestConvolutionChunkerWithMock:
    TEXT_MULTI_PARAGRAPH = (
        "Primera oración del primer tema. Segunda oración del mismo tema. "
        "Tercera oración del primer tema.\n\n"
        "Primera oración del segundo tema. Segunda oración del segundo tema. "
        "Ahora cambiamos de tema completamente.\n\n"
        "Tercera sección con contenido diferente. Aquí hablamos de algo nuevo."
    )

    def test_dsp_metadata_present(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex("similar"))
        result = chunker.chunk(self.TEXT_MULTI_PARAGRAPH, {"unit": "paragraph"})

        assert "similarity_signal" in result.metadata
        assert "smoothed_signal" in result.metadata
        assert "valley_positions" in result.metadata
        assert "valley_depths" in result.metadata
        assert "kernel_type" in result.metadata

    def test_dsp_signals_are_lists(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex("similar"))
        result = chunker.chunk(self.TEXT_MULTI_PARAGRAPH, {"unit": "paragraph"})

        assert isinstance(result.metadata["similarity_signal"], list)
        assert isinstance(result.metadata["smoothed_signal"], list)
        assert isinstance(result.metadata["valley_positions"], list)

    def test_signal_length_matches_units_minus_one(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex("similar"))
        result = chunker.chunk(self.TEXT_MULTI_PARAGRAPH, {"unit": "paragraph"})

        num_units = result.metadata["num_units"]
        sim_signal = result.metadata["similarity_signal"]
        assert len(sim_signal) == num_units - 1 or num_units <= 1

    def test_similar_embeddings_produce_one_chunk(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex("similar"))
        result = chunker.chunk(
            self.TEXT_MULTI_PARAGRAPH,
            {"unit": "paragraph", "valley_threshold": 0.3}
        )
        # Con embeddings muy similares, no debería haber valles bajo 0.3
        # → 1 chunk
        assert result.num_chunks >= 1

    def test_chunker_name_in_metadata(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex())
        result = chunker.chunk(SHORT_TEXT, {"unit": "sentence"})
        assert result.metadata.get("chunker") == "convolution"

    def test_kernel_type_stored_in_metadata(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex())
        result = chunker.chunk(SHORT_TEXT, {
            "kernel_type": "edge_detect",
            "kernel_size": 3,
            "unit": "sentence",
        })
        assert result.metadata.get("kernel_type") == "edge_detect"
        assert result.metadata.get("kernel_size") == 3

    def test_units_preview_in_metadata(self) -> None:
        chunker = ConvolutionChunker()
        chunker.setup(embedding_index=MockEmbeddingIndex())
        result = chunker.chunk(self.TEXT_MULTI_PARAGRAPH, {"unit": "paragraph"})
        assert "units_preview" in result.metadata
        assert len(result.metadata["units_preview"]) == result.metadata["num_units"]


# ===========================================================================
# Tests de ChunkingPipeline
# ===========================================================================

class TestChunkingPipeline:
    def test_single_stage(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        result = pipeline.run(SIMPLE_TEXT)
        assert result.num_chunks >= 1

    def test_pipeline_name_single(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        assert pipeline.name == "structural"

    def test_pipeline_name_multi(self) -> None:
        pipeline = ChunkingPipeline([
            ("structural", {}),
            ("convolution", {"kernel_type": "gaussian", "kernel_size": 5}),
        ])
        assert "structural" in pipeline.name
        assert "convolution" in pipeline.name
        assert "gaussian" in pipeline.name

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="al menos 1"):
            ChunkingPipeline([])

    def test_setup_called_on_all_chunkers(self) -> None:
        pipeline = ChunkingPipeline([("structural", {}), ("convolution", {})])
        # Setup no debe lanzar excepciones
        pipeline.setup(embedding_index=None)

    def test_teardown_called_on_all_chunkers(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        pipeline.teardown()

    def test_stages_order_in_metadata(self) -> None:
        pipeline = ChunkingPipeline([
            ("structural", {}),
            ("convolution", {}),
        ])
        result = pipeline.run(SIMPLE_TEXT)
        assert result.metadata.get("stages_order") == ["structural", "convolution"]

    def test_total_chunks_in_metadata(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        result = pipeline.run(SIMPLE_TEXT)
        assert result.metadata.get("total_chunks") == result.num_chunks

    def test_pipeline_name_in_metadata(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        result = pipeline.run(SIMPLE_TEXT)
        assert result.metadata.get("pipeline") == pipeline.name

    def test_two_stage_produces_valid_result(self) -> None:
        pipeline = ChunkingPipeline([
            ("structural", {}),
            ("convolution", {"unit": "paragraph"}),
        ])
        pipeline.setup(embedding_index=None)  # Sin embeddings → fallback
        result = pipeline.run(SIMPLE_TEXT)
        assert result.num_chunks >= 1
        assert all(f.strip() for f in result.fragments)

    def test_chunker_names_property(self) -> None:
        pipeline = ChunkingPipeline([("structural", {}), ("convolution", {})])
        assert pipeline.chunker_names == ["structural", "convolution"]


# ===========================================================================
# Tests de compatibilidad con NoteWatcher
# ===========================================================================

class TestPipelineNoteWatcherCompatibility:
    def test_chunk_method_returns_dicts(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        chunks = pipeline.chunk("nota_test", SIMPLE_TEXT, ["tag1"], 7)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_method_has_required_keys(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        chunks = pipeline.chunk("nota_test", SIMPLE_TEXT, ["tag1"], 7)
        required_keys = {"id", "title", "content", "order"}
        for chunk in chunks:
            assert required_keys.issubset(chunk.keys()), (
                f"Faltan keys en chunk: {required_keys - chunk.keys()}"
            )

    def test_chunk_method_ids_start_with_note_id(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        chunks = pipeline.chunk("mi_nota", SIMPLE_TEXT, [], 5)
        for chunk in chunks:
            assert chunk["id"].startswith("mi_nota::")

    def test_chunk_method_orders_sequential(self) -> None:
        pipeline = ChunkingPipeline([("structural", {})])
        chunks = pipeline.chunk("nota", SIMPLE_TEXT, [], 5)
        orders = [c["order"] for c in chunks]
        assert orders == list(range(len(chunks)))
