# Comparación de Estrategias RAG

> Generado: 2026-06-29 06:40  |  20 runs

## Tabla Comparativa

| Run ID | Chunking | Retrieval | Chunk(ms) | Ret(ms) | Tokens | RAGAS | RAGAS P | RAGAS R | E_tok |
|--------|----------|-----------|-----------|---------|--------|-------|---------|---------|-------|
| `document__embeddings` | document | embeddings | 463.1 | 24.1 | 3739 | 0.3545 | 0.2909 | 0.4091 | 202284.1 |
| `document__splay` | document | splay | **169.3** | 1.9 | 3525 | 0.2364 | 0.1636 | 0.2545 | 228389.2 |
| `document+convolution__embeddings` | document → convolution | embeddings | 4036.3 | 49.7 | 835 | 0.3591 | 0.2636 | 0.4636 | 30361.2 |
| `document+convolution__splay` | document → convolution | splay | 6699.0 | 4.6 | 831 | 0.0364 | 0.0364 | 0.0545 | 75729.7 |
| `naive_200__embeddings` | naive | embeddings | 1884.4 | 51.9 | 635 | 0.4227 | 0.3455 | 0.5000 | 26899.8 |
| `naive_200__splay` | naive | splay | 1590.4 | 4.0 | 566 | 0.1227 | 0.0591 | 0.0818 | 41851.0 |
| `naive_200+convolution__embeddings` | naive → convolution | embeddings | 9374.3 | 55.8 | 445 | 0.5818 | 0.3727 | 0.5636 | 13003.3 |
| `naive_200+convolution__splay` | naive → convolution | splay | 9954.4 | 5.1 | 583 | 0.1864 | 0.0773 | 0.1455 | 42656.3 |
| `naive_500__embeddings` | naive | embeddings | 924.7 | 52.5 | 635 | 0.4227 | 0.3455 | 0.5000 | 26899.8 |
| `naive_500__splay` | naive | splay | 863.7 | 4.2 | 566 | 0.1227 | 0.0591 | 0.0818 | 41851.0 |
| `naive_500+convolution__embeddings` | naive → convolution | embeddings | 7900.5 | 51.4 | 445 | 0.5818 | 0.3727 | 0.5636 | 13003.3 |
| `naive_500+convolution__splay` | naive → convolution | splay | 7880.1 | 4.7 | 583 | 0.1864 | 0.0773 | 0.1455 | 42656.3 |
| `structural__embeddings` | structural | embeddings | 1998.6 | 53.1 | 913 | 0.5864 | 0.3636 | **0.6182** | 25085.3 |
| `structural__splay` | structural | splay | 1681.6 | **1.7** | 821 | 0.0955 | 0.0455 | 0.1364 | 67457.3 |
| `structural+convolution__embeddings` | structural → convolution | embeddings | 3436.6 | 22.6 | 307 | 0.4818 | 0.3000 | 0.4636 | 11925.1 |
| `structural+convolution__splay` | structural → convolution | splay | 3369.4 | 1.9 | 216 | 0.0182 | 0.0182 | 0.0182 | 19729.9 |
| `semantic__embeddings` | semantic | embeddings | 7382.1 | 19.7 | 219 | 0.5909 | **0.3818** | 0.5273 | 1823.0 |
| `semantic__splay` | semantic | splay | 9920.7 | 5.1 | **165** | 0.0091 | 0.0227 | 0.0000 | 15136.4 |
| `semantic+convolution__embeddings` | semantic → convolution | embeddings | 32990.2 | 53.5 | 200 | **0.6273** | 0.3500 | 0.5273 | **1793.8** |
| `semantic+convolution__splay` | semantic → convolution | splay | 31982.3 | 4.8 | **165** | 0.0091 | 0.0227 | 0.0000 | 15136.4 |

> **Nota**: Los valores en **negrita** indican el mejor rendimiento en esa métrica.
> E_tok menor = más eficiente (menos tokens por unidad de calidad RAGAS).

---
### ⚡ Métricas de Desempeño y Costo
- **Chunk(ms)**: Tiempo en ms para fragmentar la base de conocimiento (overhead de ingesta).
- **Ret(ms)**: Tiempo en ms para recuperar información durante una query. Fundamental en sistemas embebidos.
- **Tokens**: Texto inyectado al LLM. Define el consumo de memoria VRAM.
- **RAGAS Score (0 a 1)**: Evaluación automatizada (LLM Juez) que verifica si el texto responde la pregunta.
- **E_tok ($E_{tok}$)**: Eficiencia ($Tokens / RAGAS$). **MENOR es MEJOR**. Evalúa la memoria invertida por punto de calidad.
- **RAGAS P (Context Precision)**: Evalúa si los fragmentos relevantes están bien posicionados. (LLM Evaluated).
- **RAGAS R (Context Recall)**: Evalúa si el contexto recuperado logra alinear toda la respuesta esperada. (LLM Evaluated).

### 🧠 Estrategias Evaluadas (Aportes de Tesis)
#### 1. Fases de Chunking (Segmentación)
- **document**: No divide el texto. La nota completa es un solo chunk. Sirve como línea base del peor rendimiento (satura el contexto).
- **naive**: Corta el texto estáticamente por cantidad de tokens (ej. 200, 500). El método más popular pero ignorante del contenido.
- **structural**: Divide estáticamente respetando los encabezados Markdown y párrafos. Es el método más lógico para documentos estructurados.
- **semantic**: Corta midiendo la similitud del coseno entre oraciones, creando un nuevo chunk cuando detecta un cambio brusco de tema.
- **[cualquiera] → convolution** *(Aporte)*: Toma los fragmentos de la etapa anterior y aplica filtros de procesamiento de señales (convolución) para fusionar dinámicamente aquellos que comparten contexto, mejorando la cohesión.

#### 2. Fases de Recuperación (Retrieval)
- **embeddings**: RAG tradicional. Búsqueda vectorial exhaustiva por distancia coseno. Precisión alta, latencia y costo computacional alto.
- **splay** *(Aporte)*: Caché adaptativa Splay Tree. Reorganiza accesos recientes para retornar hits en ~0ms sin pasar por inferencia ONNX, optimizando drásticamente la latencia en hardware limitado.
