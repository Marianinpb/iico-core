# iico-core

`iico-core` es la librería central del proyecto iico-agent. Proporciona un arnés (Harness) completamente agnóstico de la interfaz de usuario para gestionar la interacción con modelos de lenguaje grandes (LLMs).

## 🚀 Características

- **Independencia de Interfaz**: Funciona con cualquier UI (Textual, Open WebUI, scripts, etc.).
- **Gestión de Memoria Pasiva**: Carga automática de notas Markdown + YAML al contexto del LLM.
- **Eficiencia de Recursos**: Filtra qué contexto inyectar en el prompt mediante presupuestos de tokens.
- **Múltiples Proveedores**: Compatible de fábrica con Ollama y cualquier servicio compatible con la API de OpenAI (llama.cpp, LM Studio, vLLM).

## 📦 Instalación

Puedes instalar esta librería localmente en modo editable (ideal para desarrollo):

```bash
pip install -e .
```

Si necesitas herramientas de desarrollo y formateo:

```bash
pip install -e ".[dev]"
```

## 🛠️ Estructura del Código

- `iico_core/harness.py`: El orquestador principal que toda interfaz debe instanciar.
- `iico_core/llm_client.py`: Clientes de API unificados bajo el mismo Protocol.
- `iico_core/memory/passive.py`: El motor de la memoria pasiva determinista.
- `iico_core/types.py`: Tipos y Dataclasses compartidos (como `HarnessEvent`).

## 💻 Uso Básico

```python
import asyncio
from pathlib import Path
from iico_core import Harness, HarnessConfig, ProviderConfig, HarnessEventType

async def main():
    # 1. Configurar el proveedor y rutas
    config = HarnessConfig(
        provider=ProviderConfig(
            type="ollama", 
            endpoint="http://localhost:11434", 
            model="qwen2.5:7b"
        ),
        memory_path=Path("memory_store")
    )
    
    # 2. Inicializar el arnés
    harness = Harness(config)
    
    # 3. Procesar input y consumir eventos
    async for event in harness.process_input("¿Qué comandos tengo disponibles?"):
        if event.type == HarnessEventType.TOKEN:
            print(event.payload, end="", flush=True)
            
if __name__ == "__main__":
    asyncio.run(main())
```

---
*Desarrollado como parte del proyecto de investigación Tesis - IICO Agent.*
