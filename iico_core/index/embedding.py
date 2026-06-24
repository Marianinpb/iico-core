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

Persistencia de embeddings (Fase 4):
    - load_from_disk(): carga embeddings .npy pre-computados sin ONNX
    - build_from_chunks(): vectoriza solo chunks nuevos/modificados
    - update_chunk(): re-vectoriza un chunk individual
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import Chunk

if TYPE_CHECKING:
    from ..memory.chunk_store import ChunkStore
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
    3. Retorna chunks que superan el umbral (default 0.75)

    Fase 4 — Persistencia en disco:
        Usar load_from_disk() para cargar embeddings .npy pre-computados
        (sin ONNX) o build_from_chunks() para vectorizar solo chunks nuevos.

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
        self._notes: list[Chunk] = []
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
        Deprecated: use build_from_chunks() for chunk-based indexing or
        load_from_disk() for persistence.

        Vectoriza todas las notas y construye el índice en RAM.
        Llamar al inicio de la sesión o al recargar notas.
        """
        warnings.warn(
            "build_index() is deprecated. "
            "Use build_from_chunks() for chunk-based indexing "
            "or load_from_disk() for persistence.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not notes:
            self._notes = []
            self._embeddings = None
            return

        self._ensure_loaded()
        self._notes = []
        embeddings = []
        for note in notes:
            # Convertir PassiveNote a Chunk para almacenamiento interno
            chunk = Chunk(
                id=note.id,
                parent_note_id=note.id,
                title=note.id,
                content=note.content,
                tags=list(note.tags),
                priority=note.priority,
                order=0,
            )
            self._notes.append(chunk)
            text = f"{note.id} {' '.join(note.tags)} {note.content[:512]}"
            embeddings.append(self.vectorize(text))
        self._embeddings = np.stack(embeddings)  # shape: (n, 384)

    def update(self, note: "PassiveNote") -> None:
        """
        Agrega o actualiza una nota en el índice sin rebuild completo.
        Útil cuando el LLM crea una nota nueva via PassiveMemory.add_note().

        Deprecated: use update_chunk() for chunk-based updates.
        """
        self._ensure_loaded()
        text = f"{note.id} {' '.join(note.tags)} {note.content[:512]}"
        embedding = self.vectorize(text)

        # Convertir PassiveNote a Chunk para almacenamiento interno
        chunk = Chunk(
            id=note.id,
            parent_note_id=note.id,
            title=note.id,
            content=note.content,
            tags=list(note.tags),
            priority=note.priority,
            order=0,
        )

        # Buscar si ya existe
        for i, existing in enumerate(self._notes):
            if existing.id == note.id:
                self._notes[i] = chunk
                if self._embeddings is not None:
                    self._embeddings[i] = embedding
                return

        # Nota nueva: agregar
        self._notes.append(chunk)
        if self._embeddings is None:
            self._embeddings = embedding[np.newaxis, :]
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    # ------------------------------------------------------------------
    # Persistencia en disco (Fase 4)
    # ------------------------------------------------------------------

    def load_from_disk(
        self, chunks: list[Chunk], chunk_store: "ChunkStore | None" = None,
    ) -> bool:
        """Carga embeddings pre-computados desde archivos .npy en disco.

        Recorre la lista de chunks: para cada uno, intenta cargar su
        embedding .npy (via chunk.embedding_path o chunk_store.load_embedding).
        No llama a ONNX en ningún momento — carga pura de numpy.

        Args:
            chunks: lista de chunks con sus metadatos.
            chunk_store: si se provee, usa su load_embedding() (con caché)
                         en vez de np.load directo.

        Returns:
            True si al menos un embedding se cargó exitosamente.
            False si ningún chunk tenía embedding en disco (se necesita
            build_from_chunks).
        """
        self._notes = []
        embeddings: list[np.ndarray] = []
        any_loaded = False

        for chunk in chunks:
            try:
                if chunk_store is not None:
                    emb = chunk_store.load_embedding(chunk)
                else:
                    if chunk.embedding_path is None or not chunk.embedding_path.exists():
                        raise FileNotFoundError(
                            f"No embedding .npy for chunk {chunk.id}"
                        )
                    emb = np.load(str(chunk.embedding_path))
                self._notes.append(chunk)
                embeddings.append(emb)
                any_loaded = True
            except FileNotFoundError:
                # Chunk sin embedding en disco: se omite silenciosamente
                continue
            except ImportError:
                # numpy no instalado: propagar
                raise

        if embeddings:
            self._embeddings = np.stack(embeddings)
        else:
            self._embeddings = None

        return any_loaded

    def build_from_chunks(
        self, chunks: list[Chunk], force: bool = False,
    ) -> None:
        """Construye el índice a partir de chunks, vectorizando solo los necesarios.

        Para cada chunk:
        - Si force=True o chunk no tiene embedding_path → vectoriza con ONNX
          y guarda el .npy en chunk.embedding_path.
        - Si force=False y el chunk ya tiene embedding_path → omite.

        Args:
            chunks: lista de chunks (con o sin embeddings previos).
            force: si True, re-vectoriza todos los chunks.
        """
        if not chunks:
            self._notes = []
            self._embeddings = None
            return

        self._ensure_loaded()
        self._notes = []
        embeddings: list[np.ndarray] = []

        for chunk in chunks:
            # Determinar si necesita vectorización
            needs_vectorize = force or chunk.embedding_path is None
            if not needs_vectorize and chunk.embedding_path is not None:
                needs_vectorize = not chunk.embedding_path.exists()

            if needs_vectorize:
                text = self._chunk_text(chunk)
                embedding = self.vectorize(text)

                # Guardar embedding en disco si hay ruta
                if chunk.embedding_path is not None:
                    chunk.embedding_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(chunk.embedding_path), embedding)
                embeddings.append(embedding)
            else:
                # Cargar desde disco (ya existe el .npy)
                emb = np.load(str(chunk.embedding_path))
                embeddings.append(emb)

            self._notes.append(chunk)

        if embeddings:
            self._embeddings = np.stack(embeddings)
        else:
            self._embeddings = None

    def update_chunk(self, chunk: Chunk) -> None:
        """Re-vectoriza un chunk individual y actualiza la fila en el índice.

        - Si el chunk ya existe en self._notes (por id), reemplaza su fila
          en self._embeddings y sobreescribe el .npy en disco.
        - Si el chunk no existe en el índice, lo agrega al final.

        Args:
            chunk: el chunk a actualizar o insertar.
        """
        self._ensure_loaded()
        text = self._chunk_text(chunk)
        embedding = self.vectorize(text)

        # Guardar en disco si hay ruta
        if chunk.embedding_path is not None:
            chunk.embedding_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(chunk.embedding_path), embedding)

        # Buscar si ya existe por id
        for i, existing in enumerate(self._notes):
            if existing.id == chunk.id:
                self._notes[i] = chunk
                if self._embeddings is not None:
                    self._embeddings[i] = embedding
                return

        # Chunk nuevo: agregar al final
        self._notes.append(chunk)
        if self._embeddings is None:
            self._embeddings = embedding[np.newaxis, :]
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

    @staticmethod
    def _chunk_text(chunk: Chunk) -> str:
        """Construye el texto a vectorizar para un chunk."""
        return f"{chunk.id} {' '.join(chunk.tags)} {chunk.content[:512]}"

    # ------------------------------------------------------------------
    # Búsqueda semántica
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        threshold: float = 0.75,
        top_k: int = 5,
    ) -> "list[tuple[Chunk, float]]":
        """
        Busca chunks semánticamente similares al query.

        Args:
            query: texto a buscar (se vectoriza internamente)
            threshold: umbral mínimo de similitud del coseno (0.0 - 1.0)
            top_k: número máximo de resultados

        Returns:
            Lista de (chunk, score) ordenada por score descendente,
            solo incluyendo chunks con score >= threshold.
        """
        if not self._notes or self._embeddings is None:
            return []

        query_embedding = self.vectorize(query)

        # Cosine similarity de una vez (vectorizado con numpy)
        # Embeddings ya normalizados → dot product = cosine
        scores = self._embeddings @ query_embedding  # shape: (n,)

        # Filtrar por umbral y ordenar
        results: list[tuple[Chunk, float]] = []
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
