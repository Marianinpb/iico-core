# iico-core

`iico-core` es la librería central del proyecto iico-agent. Proporciona un arnés (Harness) completamente agnóstico de la interfaz de usuario para gestionar la interacción con modelos de lenguaje grandes (LLMs).

## 🚀 Características Principales

- **Independencia de Interfaz**: El núcleo funciona con cualquier UI (Textual, Open WebUI, scripts, etc.), ya que expone métodos genéricos asíncronos y eventos estructurados (`HarnessEvent`).
- **Memoria Pasiva Dual**: Carga automática de notas Markdown + YAML al contexto del LLM. Combina una *Caché Splay Tree* (Nivel 2) de altísima velocidad y una búsqueda local por *Embeddings Semánticos* (Nivel 1, usando ONNX) para inyectar contexto preciso sin depender de la nube y sin saturar la VRAM de la GPU.
- **Registro de Skills y Bucle Autónomo (ReAct)**: Soporte nativo para herramientas externas. El agente utiliza un paradigma *Spec-Driven Development* interno para formular planes estructurados, resolver topológicamente dependencias de las tareas y usar el `ShellBridge` para ejecutar comandos de forma autónoma.
- **Múltiples Proveedores**: Compatible de fábrica con Ollama y cualquier servicio compatible con la API de OpenAI (llama.cpp, LM Studio, vLLM) gracias a una interfaz de cliente de LLM unificada.

## 🏗️ Arquitectura del Arnés (Harness)

La arquitectura principal de la librería está dividida en submódulos especializados para lograr el desacoplamiento:

- **`iico_core.harness`**: Contiene la clase orquestadora principal `Harness`, encargada de ensamblar dinámicamente el *System Prompt* y coordinar el ciclo de razonamiento (ReAct) con los eventos de salida.
- **`iico_core.llm_client`**: Interfaces unificadas tras un *Protocol* que oculta las particularidades de si nos conectamos por Ollama local o a través de una API de OpenAI remota.
- **`iico_core.memory`**: Módulos responsables de la memoria del agente. `passive.py` carga la base de conocimiento local (Markdown+YAML), mientras que `active.py` (`SkillRegistry`) cataloga las herramientas disponibles.
- **`iico_core.index`**: Algoritmos de indexación. Destacan `embedding.py` (motor semántico de Nivel 1) y `splay_tree.py` (Caché O(1) de Nivel 2 basada en auto-balances).
- **`iico_core.bridge`**: Puente seguro (`ShellBridge`) para ejecutar comandos del sistema que el agente decide utilizar.
- **`iico_core.types`**: Dataclasses unificadas (como `ChatMessage`, `HarnessConfig`, `HarnessEvent`) que actúan como la "API pública".

## 📦 Instalación

Puedes instalar esta librería localmente en modo editable, lo cual es ideal para seguir el desarrollo de la tesis:

```bash
pip install -e .
```

Si necesitas herramientas de desarrollo (testing, formatters, etc.):

```bash
pip install -e ".[dev]"
```

## 💻 Uso Básico

A continuación, se presenta un ejemplo robusto de inicialización, configurando un proveedor de modelo y la ruta de la memoria pasiva:

```python
import asyncio
from pathlib import Path
from iico_core import Harness, HarnessConfig, ProviderConfig, HarnessEventType

async def main():
    # 1. Configurar el proveedor y rutas (ej. Ollama con el modelo Qwen)
    config = HarnessConfig(
        provider=ProviderConfig(
            type="ollama", 
            endpoint="http://localhost:11434", 
            model="qwen2.5:7b"
        ),
        memory_path=Path("memory_store")
    )
    
    # 2. Inicializar el arnés, el cual carga pasivamente el contexto
    harness = Harness(config)
    
    # 3. Procesar input y consumir eventos reactivos
    async for event in harness.process_input("¿Qué comandos tengo disponibles en tu memoria?"):
        if event.type == HarnessEventType.TOKEN:
            # Imprimir el stream de tokens al stdout en tiempo real
            print(event.payload, end="", flush=True)
            
if __name__ == "__main__":
    asyncio.run(main())
```

---
*Desarrollado como parte del proyecto de investigación Tesis - IICO Agent.*
