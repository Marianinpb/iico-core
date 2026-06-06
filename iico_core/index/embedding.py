"""
iico_core/index/embedding.py
=============================
Índice de Búsqueda Semántica — Nivel 1 de la arquitectura de memoria dual.

Usa el modelo 'all-MiniLM-L6-v2' exportado a ONNX (~80MB) corriendo 100%
en CPU con onnxruntime. Sin dependencia de PyTorch.

Flujo en la arquitectura de dos niveles:
    Splay Tree (Nivel 2) → miss → EmbeddingIndex.search() → resultados
    → insertar en Splay para caché futura

Formato del modelo:
    - Input: token IDs + attention mask + token type IDs
    - Output: embeddings de dimensión 384
    - Normalización: L2 (para que cosine similarity = dot product)

Descarga automática del modelo desde HuggingFace Hub la primera vez.
Se cachea en ~/.cache/iico/models/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..memory.passive import PassiveNote


# Directorio de caché de modelos
_DEFAULT_CACHE = Path.home() / ".cache" / "iico" / "models"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _require_onnxruntime() -> Any:
    """Importa onnxruntime o lanza un error claro con instrucciones."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        raise ImportError(
            "onnxruntime no está instalado. Para habilitar la búsqueda semántica:\n"
            "  pip install iico-core[embeddings]\n"
            "  o bien: pip install onnxruntime numpy tokenizers"
        ) from None


def _require_tokenizers() -> Any:
    try:
        from tokenizers import Tokenizer
        return Tokenizer
    except ImportError:
        raise ImportError(
            "tokenizers no está instalado. Para habilitar la búsqueda semántica:\n"
            "  pip install iico-core[embeddings]"
        ) from None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud del coseno entre dos vectores normalizados."""
    # Si los vectores ya están normalizados (L2=1), el coseno = dot product
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _download_model(cache_dir: Path) -> tuple[Path, Path]:
    """
    Descarga el modelo ONNX y el tokenizador desde HuggingFace Hub.
    Retorna (model_path, tokenizer_path).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub no está instalado. Instala con:\n"
            "  pip install huggingface-hub"
        ) from None

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cache_dir / "all-MiniLM-L6-v2"
    model_dir.mkdir(exist_ok=True)

    model_path = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="onnx/model.onnx",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    tokenizer_path = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="tokenizer.json",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    return Path(model_path), Path(tokenizer_path)


class EmbeddingIndex:
    """
    Índice semántico vectorial (Nivel 1 de la arquitectura de memoria dual).

    Flujo:
    1. Al iniciar: vectoriza todas las notas y guarda embeddings en RAM
    2. En search(): vectoriza el query y calcula cosine similarity
    3. Retorna notas que superan el umbral (default 0.75)

    Sinergia con Splay Tree:
    - Los resultados de search() se insertan en el Splay Tree (Nivel 2)
    - Las búsquedas subsecuentes del mismo tema las resuelve el Splay
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        model_path: Path | str | None = None,
        tokenizer_path: Path | str | None = None,
    ):
        """
        Args:
            cache_dir: directorio de caché de modelos (default: ~/.cache/iico/models)
            model_path: ruta explícita al .onnx (omite descarga si se provee)
            tokenizer_path: ruta explícita al tokenizer.json
        """
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self._model_path = Path(model_path) if model_path else None
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else None

        self._session = None          # onnxruntime.InferenceSession
        self._tokenizer = None        # tokenizers.Tokenizer
        self._notes: list["PassiveNote"] = []
        self._embeddings: np.ndarray | None = None  # shape: (n_notes, 384)
        self._loaded = False

    # ------------------------------------------------------------------
    # Carga del modelo (lazy)
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Carga el modelo ONNX y el tokenizador si no están cargados aún."""
        if self._loaded:
            return

        ort = _require_onnxruntime()
        Tokenizer = _require_tokenizers()

        # Resolver rutas del modelo
        if self._model_path is None or self._tokenizer_path is None:
            model_path, tokenizer_path = self._resolve_model_paths()
        else:
            model_path = self._model_path
            tokenizer_path = self._tokenizer_path

        # Cargar sesión ONNX (CPU)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2  # Conservar CPU para el LLM
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Cargar tokenizador
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self._tokenizer.enable_truncation(max_length=128)

        self._loaded = True

    def _resolve_model_paths(self) -> tuple[Path, Path]:
        """Busca el modelo en caché local; lo descarga si no existe."""
        model_dir = self._cache_dir / "all-MiniLM-L6-v2"
        model_path = model_dir / "onnx" / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if model_path.exists() and tokenizer_path.exists():
            return model_path, tokenizer_path

        print("[EmbeddingIndex] Descargando modelo all-MiniLM-L6-v2 (~80MB)...")
        return _download_model(self._cache_dir)

    # ------------------------------------------------------------------
    # Vectorización
    # ------------------------------------------------------------------

    def vectorize(self, text: str) -> np.ndarray:
        """
        Genera el embedding de un texto.
        Retorna un vector numpy de dimensión 384, normalizado (L2=1).
        """
        self._ensure_loaded()
        encoding = self._tokenizer.encode(text)
        input_ids      = np.array([encoding.ids],           dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids],       dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # Mean pooling sobre los token embeddings
        token_embeddings = outputs[0]  # shape: (1, seq_len, 384)
        mask = attention_mask[0].astype(np.float32)
        embedding = np.sum(token_embeddings[0] * mask[:, np.newaxis], axis=0) / mask.sum()

        # Normalizar a L2=1 para que cosine similarity = dot product
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Gestión del índice
    # ------------------------------------------------------------------

    def build_index(self, notes: "list[PassiveNote]") -> None:
        """
        Vectoriza todas las notas y construye el índice en RAM.
        Llamar al inicio de la sesión o al recargar notas.
        """
        if not notes:
            self._notes = []
            self._embeddings = None
            return

        self._ensure_loaded()
        self._notes = list(notes)
        embeddings = []
        for note in self._notes:
            text = f"{note.id} {' '.join(note.tags)} {note.content[:512]}"
            embeddings.append(self.vectorize(text))
        self._embeddings = np.stack(embeddings)  # shape: (n, 384)

    def update(self, note: "PassiveNote") -> None:
        """
        Agrega o actualiza una nota en el índice sin rebuild completo.
        Útil cuando el LLM crea una nota nueva via PassiveMemory.add_note().
        """
        self._ensure_loaded()
        text = f"{note.id} {' '.join(note.tags)} {note.content[:512]}"
        embedding = self.vectorize(text)

        # Buscar si ya existe
        for i, existing in enumerate(self._notes):
            if existing.id == note.id:
                self._notes[i] = note
                if self._embeddings is not None:
                    self._embeddings[i] = embedding
                return

        # Nota nueva: agregar
        self._notes.append(note)
        if self._embeddings is None:
            self._embeddings = embedding[np.newaxis, :]
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    # ------------------------------------------------------------------
    # Búsqueda semántica
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        threshold: float = 0.75,
        top_k: int = 5,
    ) -> "list[tuple[PassiveNote, float]]":
        """
        Busca notas semánticamente similares al query.

        Args:
            query: texto a buscar (se vectoriza internamente)
            threshold: umbral mínimo de similitud del coseno (0.0 - 1.0)
            top_k: número máximo de resultados

        Returns:
            Lista de (nota, score) ordenada por score descendente,
            solo incluyendo notas con score >= threshold.
        """
        if not self._notes or self._embeddings is None:
            return []

        query_embedding = self.vectorize(query)

        # Cosine similarity de una vez (vectorizado con numpy)
        # Embeddings ya normalizados → dot product = cosine
        scores = self._embeddings @ query_embedding  # shape: (n,)

        # Filtrar por umbral y ordenar
        results: list[tuple["PassiveNote", float]] = []
        for i, score in enumerate(scores):
            if score >= threshold:
                results.append((self._notes[i], float(score)))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    # ------------------------------------------------------------------
    # Inspección
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def index_size(self) -> int:
        return len(self._notes)
