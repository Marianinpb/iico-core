"""
tests/test_chunker_structural.py
=================================
Tests unitarios del Chunker estructural de Markdown.

Verifica que el Chunker divida correctamente una PassiveNote en Chunks
basándose en encabezados ATX, bloques de código, reglas horizontales
y límites de tokens.
"""

import hashlib
import pytest

from iico_core.memory.chunker import Chunker
from iico_core.memory.passive import PassiveNote


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker() -> Chunker:
    return Chunker(max_chunk_tokens=512)


@pytest.fixture
def small_chunker() -> Chunker:
    """Chunker con límite bajo para forzar división por párrafos."""
    return Chunker(max_chunk_tokens=20)


@pytest.fixture
def note_with_headers() -> PassiveNote:
    return _make_note(
        note_id="arquitectura",
        tags=["arquitectura", "base"],
        priority=7,
        content="## Intro\n\nPresentacion del sistema.\n\n"
                 "## Diseno\n\nArquitectura de componentes.\n\n"
                 "## Implementacion\n\nDetalles del codigo.\n",
    )


@pytest.fixture
def note_with_subsections() -> PassiveNote:
    return _make_note(
        note_id="algoritmos",
        tags=["algoritmos"],
        priority=5,
        content="## Splay Tree\n\n"
                 "El splay tree es auto-balanceable.\n\n"
                 "### Rotaciones\n\n"
                 "Zig, zig-zig y zig-zag.\n\n"
                 "### Complejidad\n\n"
                 "Amortizada O(log n).\n\n"
                 "## AVL\n\n"
                 "Balance estricto por altura.\n",
    )


@pytest.fixture
def note_with_code_blocks() -> PassiveNote:
    return _make_note(
        note_id="scripting",
        tags=["python", "shell"],
        priority=6,
        content="## Instalacion\n\n"
                 "Ejecuta el siguiente comando:\n\n"
                 "```bash\n"
                 "pip install iico-core\n"
                 "uv sync\n"
                 "```\n\n"
                 "## Script Python\n\n"
                 "```python\n"
                 "def main():\n"
                 "    print('hola')\n"
                 "```\n",
    )


@pytest.fixture
def note_with_horizontal_rules() -> PassiveNote:
    return _make_note(
        note_id="separado",
        tags=["docs"],
        priority=4,
        content="## Seccion A\n\nContenido A.\n\n"
                 "---\n\n"
                 "## Seccion B\n\nContenido B.\n\n"
                 "***\n\n"
                 "## Seccion C\n\nContenido C.\n",
    )


@pytest.fixture
def oversized_note() -> PassiveNote:
    """Nota con una sección que excede el límite de tokens de small_chunker."""
    return _make_note(
        note_id="largo",
        tags=["extenso"],
        priority=3,
        content="## Intro\n\n"
                 + "Parrafo A con suficiente texto para superar el limite. " * 10 + "\n\n"
                 + "Parrafo B tambien con mucho contenido textual aqui. " * 10 + "\n\n"
                 + "Parrafo C igualmente extenso con palabras largas. " * 10 + "\n",
    )


@pytest.fixture
def note_with_h1() -> PassiveNote:
    """Nota con H1 (que NO debe generar chunk separado)."""
    return _make_note(
        note_id="h1-test",
        tags=["markdown"],
        priority=5,
        content="# Titulo Principal\n\n"
                 "Texto introductorio antes de las secciones.\n\n"
                 "## Primera Seccion\n\n"
                 "Contenido de la primera seccion.\n",
    )


# ---------------------------------------------------------------------------
# Test: headers ## → número correcto de chunks
# ---------------------------------------------------------------------------

def test_chunks_count_with_headers(chunker, note_with_headers):
    """Tres encabezados ## deben producir 3 chunks."""
    chunks = chunker.chunk_note(note_with_headers)
    assert len(chunks) == 3


def test_chunks_have_sequential_order(chunker, note_with_headers):
    """Los chunks deben tener orden secuencial 0, 1, 2."""
    chunks = chunker.chunk_note(note_with_headers)
    orders = [c.order for c in chunks]
    assert orders == [0, 1, 2]


def test_chunks_titles_match_headers(chunker, note_with_headers):
    """Los títulos de los chunks deben coincidir con los encabezados ##."""
    chunks = chunker.chunk_note(note_with_headers)
    titles = [c.title for c in chunks]
    assert titles == ["Intro", "Diseno", "Implementacion"]


# ---------------------------------------------------------------------------
# Test: sub-encabezados ### → jerarquía
# ---------------------------------------------------------------------------

def test_subsections_merge_when_fit(chunker, note_with_subsections):
    """Los ### bajo un ## deben fusionarse si caben en el límite."""
    chunks = chunker.chunk_note(note_with_subsections)
    # ## Splay Tree + ### Rotaciones + ### Complejidad → un chunk fusionado
    # ## AVL → otro chunk
    # Total esperado: 2 chunks
    assert len(chunks) == 2


def test_subsections_not_merged_when_oversized(small_chunker, note_with_subsections):
    """Con límite bajo, cada ### debe ser su propio chunk."""
    chunks = small_chunker.chunk_note(note_with_subsections)
    # Deberían ser al menos 3 chunks (cada sección por separado)
    assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# Test: bloques de código
# ---------------------------------------------------------------------------

def test_code_blocks_extracted_as_chunks(chunker, note_with_code_blocks):
    """Los bloques de código deben ser chunks independientes."""
    chunks = chunker.chunk_note(note_with_code_blocks)
    code_chunks = [c for c in chunks if c.title.startswith("codigo")]
    assert len(code_chunks) == 2


def test_code_block_has_language_in_title(chunker, note_with_code_blocks):
    """El título del chunk de código debe incluir el lenguaje."""
    chunks = chunker.chunk_note(note_with_code_blocks)
    code_titles = [c.title for c in chunks if c.title.startswith("codigo")]
    assert "codigo bash" in code_titles
    assert "codigo python" in code_titles


def test_code_block_content_preserved(chunker, note_with_code_blocks):
    """El contenido del bloque de código debe preservarse."""
    chunks = chunker.chunk_note(note_with_code_blocks)
    python_chunk = next(c for c in chunks if c.title == "codigo python")
    assert "def main():" in python_chunk.content
    assert "print('hola')" in python_chunk.content


# ---------------------------------------------------------------------------
# Test: reglas horizontales
# ---------------------------------------------------------------------------

def test_horizontal_rules_separate_sections(chunker, note_with_horizontal_rules):
    """Las reglas horizontales deben actuar como separadores de chunks."""
    chunks = chunker.chunk_note(note_with_horizontal_rules)
    # Tres secciones ##A, ##B, ##C = 3 chunks
    assert len(chunks) == 3
    titles = [c.title for c in chunks]
    assert titles == ["Seccion A", "Seccion B", "Seccion C"]


# ---------------------------------------------------------------------------
# Test: secciones sobredimensionadas
# ---------------------------------------------------------------------------

def test_oversized_section_splits_by_paragraphs(small_chunker, oversized_note):
    """Sección grande debe dividirse en múltiples chunks (parte 1, parte 2...)."""
    chunks = small_chunker.chunk_note(oversized_note)
    assert len(chunks) > 1
    parte_titles = [c.title for c in chunks if "(parte" in c.title]
    assert len(parte_titles) >= 1


def test_oversized_section_sequential_order(small_chunker, oversized_note):
    """Los sub-chunks de una sección grande deben mantener orden secuencial."""
    chunks = small_chunker.chunk_note(oversized_note)
    orders = [c.order for c in chunks]
    assert orders == sorted(orders)
    assert orders == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Test: formato de IDs
# ---------------------------------------------------------------------------

def test_chunk_id_format(chunker, note_with_headers):
    """ID debe seguir el formato parent_note_id::slug-del-titulo."""
    chunks = chunker.chunk_note(note_with_headers)
    for chunk in chunks:
        assert chunk.id.startswith("arquitectura::")
        assert "::" in chunk.id
        # El slug no debe tener espacios ni mayúsculas
        slug_part = chunk.id.split("::", 1)[1]
        assert " " not in slug_part
        assert slug_part == slug_part.lower()


def test_chunk_id_slugifies_special_chars(chunker):
    """Caracteres especiales en el título deben slugificarse correctamente."""
    note = _make_note(
        note_id="test",
        content="## Rotaciones Splay (zig, zig-zag)\n\nContenido.\n",
    )
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].id == "test::rotaciones-splay-zig-zig-zag"


def test_chunk_id_slugifies_tildes(chunker):
    """Las tildes deben eliminarse del slug (solo alfanumérico)."""
    note = _make_note(
        note_id="memoria",
        content="## Árbol de Decisión\n\nContenido.\n",
    )
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].id == "memoria::rbol-de-decisin"


# ---------------------------------------------------------------------------
# Test: herencia de tags y prioridad
# ---------------------------------------------------------------------------

def test_tags_inherited_from_parent(chunker, note_with_headers):
    """Cada chunk debe heredar los tags de la nota padre."""
    chunks = chunker.chunk_note(note_with_headers)
    for chunk in chunks:
        assert chunk.tags == ["arquitectura", "base"]


def test_priority_inherited_from_parent(chunker, note_with_headers):
    """Cada chunk debe heredar la prioridad de la nota padre."""
    chunks = chunker.chunk_note(note_with_headers)
    for chunk in chunks:
        assert chunk.priority == 7


def test_parent_note_id_set_correctly(chunker, note_with_headers):
    """Cada chunk debe referenciar correctamente a su nota padre."""
    chunks = chunker.chunk_note(note_with_headers)
    for chunk in chunks:
        assert chunk.parent_note_id == "arquitectura"


# ---------------------------------------------------------------------------
# Test: content_hash
# ---------------------------------------------------------------------------

def test_content_hash_is_sha256(chunker, note_with_headers):
    """content_hash debe ser el SHA-256 del contenido del chunk."""
    chunks = chunker.chunk_note(note_with_headers)
    for chunk in chunks:
        expected = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        assert chunk.content_hash == expected


def test_content_hash_different_for_different_content(chunker, note_with_headers):
    """Chunks con distinto contenido deben tener distinto hash."""
    chunks = chunker.chunk_note(note_with_headers)
    hashes = {c.content_hash for c in chunks}
    assert len(hashes) == len(chunks)


# ---------------------------------------------------------------------------
# Test: contenido sin encabezados
# ---------------------------------------------------------------------------

def test_single_paragraph_no_headers(chunker):
    """Contenido sin encabezados → un solo chunk 'introduccion'."""
    note = _make_note(content="Solo un parrafo sin estructura.\n")
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].title == "introduccion"


def test_h1_not_treated_as_section(chunker, note_with_h1):
    """H1 (#) no debe generar chunks separados."""
    chunks = chunker.chunk_note(note_with_h1)
    # El H1 y el texto introductorio son parte del chunk "introduccion"
    # Luego viene ## Primera Seccion → chunk separado
    assert len(chunks) == 2
    assert chunks[0].title == "introduccion"
    assert "Titulo Principal" in chunks[0].content
    assert chunks[1].title == "Primera Seccion"


# ---------------------------------------------------------------------------
# Test: contenido vacío
# ---------------------------------------------------------------------------

def test_empty_content_produces_one_chunk(chunker):
    """Contenido vacío debe producir un chunk 'introduccion' vacío."""
    note = _make_note(content="")
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].title == "introduccion"
    assert chunks[0].content == ""


def test_whitespace_only_content_produces_one_chunk(chunker):
    """Contenido solo con espacios debe producir un chunk vacío."""
    note = _make_note(content="   \n\n  \n")
    chunks = chunker.chunk_note(note)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Test: mezcla de todos los elementos
# ---------------------------------------------------------------------------

def test_mixed_content_all_elements(chunker):
    """Contenido con encabezados, código, reglas y texto plano."""
    note = _make_note(
        note_id="mixto",
        tags=["todo"],
        priority=5,
        content="Texto introductorio.\n\n"
                 "## Configuracion\n\n"
                 "Archivo de config:\n\n"
                 "```yaml\n"
                 "key: value\n"
                 "```\n\n"
                 "---\n\n"
                 "## Uso\n\n"
                 "### Basico\n\n"
                 "Uso básico.\n\n"
                 "### Avanzado\n\n"
                 "Uso avanzado.\n",
    )
    chunks = chunker.chunk_note(note)
    # Esperamos: introduccion, codigo yaml, ## Configuracion (separado por ---? No, --- es separador),
    # ## Uso (con ### Basico + ### Avanzado fusionados)
    # Orden real: introduccion, codigo yaml, Configuracion (después de ---), Uso (fusionado)
    assert len(chunks) >= 3
    # Verificar que hay un chunk de código yaml
    code_chunks = [c for c in chunks if "codigo" in c.title]
    assert len(code_chunks) == 1
    # Verificar que ## Uso fusiona sus sub-secciones
    uso_chunk = next((c for c in chunks if c.title == "Uso"), None)
    assert uso_chunk is not None
    assert "### Basico" in uso_chunk.content
    assert "### Avanzado" in uso_chunk.content


# ---------------------------------------------------------------------------
# Test: _estimate_tokens
# ---------------------------------------------------------------------------

def test_estimate_tokens_returns_quarter_of_length(chunker):
    """_estimate_tokens debe devolver len(text) // 4."""
    assert chunker._estimate_tokens("1234") == 1
    assert chunker._estimate_tokens("12345678") == 2
    assert chunker._estimate_tokens("123") == 0
    assert chunker._estimate_tokens("") == 0


# ---------------------------------------------------------------------------
# Test: _slugify
# ---------------------------------------------------------------------------

def test_slugify_lowercases(chunker):
    assert chunker._slugify("HOLA") == "hola"


def test_slugify_replaces_spaces(chunker):
    assert chunker._slugify("hola mundo") == "hola-mundo"


def test_slugify_removes_special_chars(chunker):
    assert chunker._slugify("hola (mundo)!") == "hola-mundo"


def test_slugify_collapses_hyphens(chunker):
    assert chunker._slugify("hola---mundo") == "hola-mundo"


def test_slugify_strips_leading_trailing_hyphens(chunker):
    assert chunker._slugify("-hola-") == "hola"


def test_slugify_empty_string(chunker):
    assert chunker._slugify("(())") == "sin-titulo"


# ---------------------------------------------------------------------------
# Test: _generate_chunk_id
# ---------------------------------------------------------------------------

def test_generate_chunk_id_combines_parent_and_slug(chunker):
    cid = chunker._generate_chunk_id("parent", "Mi Titulo")
    assert cid == "parent::mi-titulo"


# ---------------------------------------------------------------------------
# Test: orden preservado con mezcla compleja
# ---------------------------------------------------------------------------

def test_order_preserved_across_all_chunk_types(chunker):
    """El orden debe ser secuencial incluso con mezcla de tipos de chunk."""
    note = _make_note(
        note_id="seq",
        content="## A\n\nContenido A.\n\n"
                 "```py\nprint(1)\n```\n\n"
                 "## B\n\nContenido B.\n\n"
                 "---\n\n"
                 "## C\n\nContenido C.\n",
    )
    chunks = chunker.chunk_note(note)
    orders = [c.order for c in chunks]
    assert orders == list(range(len(chunks)))
    # Sin huecos
    assert min(orders) == 0
    assert max(orders) == len(chunks) - 1
