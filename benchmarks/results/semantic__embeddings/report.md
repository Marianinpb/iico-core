# Benchmark Report — semantic__embeddings

> Generado: 2026-06-29 06:39

## Configuración

| Campo | Valor |
|-------|-------|
| Run ID | `semantic__embeddings` |
| Chunking | `semantic` |
| Retrieval | `embeddings` |
| Top-K | 5 |
| RAGAS | ✅ habilitado |
| Queries evaluadas | 11 |
| Tiempo total | 7634 ms |

### Performance

| Métrica | Valor |
|---------|-------|
| Latencia promedio | 19.67 ms |
| Tokens de contexto promedio | 219 |
| RAGAS Score promedio | 0.5909 |
| RAGAS P (Context Precision) | 0.3818 |
| RAGAS R (Context Recall) | 0.5273 |
| E_tok promedio | **1823.0** |
| RAGAS cache hit rate | 100.00% |

## Resultados por Query

| ID | Query | Lat(ms) | Tokens | RAGAS | RAGAS P | RAGAS R |
|----|-------|---------|--------|-------|---------|---------|
| `q_hw_01` | ¿Qué consideraciones de interfaz se deben tener al... | 17.9 | 165 | 0.1000 | 0.2500 | 0.0000 |
| `q_hw_02` | ¿Cuál es la mejor estructura para evitar condicion... | 25.0 | 113 | 0.8500 | 0.2500 | 0.2000 |
| `q_hw_03` | ¿Cómo se previene la desincronización al combinar ... | 19.4 | 235 | 0.7000 | 0.6000 | 0.6000 |
| `q_ai_01` | ¿Por qué es necesario dividir por la raíz cuadrada... | 17.4 | 127 | 0.0000 | 0.0000 | 0.0000 |
| `q_ai_02` | ¿Cómo soluciona la arquitectura Transformer la imp... | 19.7 | 220 | 0.6000 | 0.6000 | 0.6000 |
| `q_ai_03` | ¿Qué técnica se utiliza para evitar que el pipelin... | 17.3 | 168 | 0.9000 | 0.2500 | 0.7000 |
| `q_ai_04` | ¿Qué debe hacer un agente si se da cuenta de que l... | 20.5 | 235 | 0.5500 | 0.2500 | 0.9000 |
| `q_phys_01` | ¿Cómo determina el algoritmo de perturbación y obs... | 19.9 | 319 | 0.8500 | 0.6000 | 0.9000 |
| `q_phys_02` | ¿Cuál es la implicación física del término de corr... | 16.1 | 375 | 0.9000 | 0.6000 | 0.9000 |
| `q_phys_03` | ¿Por qué la solución analítica de las Ecuaciones d... | 25.9 | 204 | 0.0500 | 0.2000 | 0.0000 |
| `q_phys_04` | ¿Cuál es el compromiso (trade-off) de utilizar un ... | 17.3 | 245 | 1.0000 | 0.6000 | 1.0000 |

---
### ⚡ Métricas de Desempeño y Costo
- **Latencia (ms)**: Tiempo en ms para buscar información. Fundamental en sistemas embebidos.
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
