# Pipeline de ingesta a Azure AI Search (vectorial)

Este repositorio contiene un pipeline en Python que ingiere documentos (PDF/Word) desde un Blob Storage, extrae texto con Document Intelligence (OCR), obtiene embeddings desde un modelo desplegado en Microsoft Foundry y guarda documentos con embeddings en un índice vectorial de Azure AI Search.

Características:

- Autenticación: Managed Identity (DefaultAzureCredential)
- Storage: Azure Blob Storage
- OCR: Azure Document Intelligence
- Vector DB: Azure AI Search (vector search)
- Model provider: Microsoft Foundry (endpoint configurado)

Archivos principales:

- `src/main.py`: Orquestador
- `src/blob_utils.py`: Conexión a Blob Storage
- `src/ocr_foundry.py`: Lógica OCR y llamada a Foundry para embeddings
- `src/search_indexer.py`: Creación de índice y upsert usando AAD
- `src/config.py`: Variables de configuración cargadas desde entorno

Instalación:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Variables de entorno (ver `.env.example`):

- `STORAGE_ACCOUNT_NAME`, `BLOB_CONTAINER_NAME`
- `SEARCH_ENDPOINT` (p.ej. https://<my-search>.search.windows.net)
- `SEARCH_INDEX_NAME`
- `FOUNDRY_ENDPOINT` (URL para inferencia en Foundry)
- `FOUNDRY_MODEL_ID`
- `EMBEDDING_DIM` (entero, dimensión del embedding)

Uso rápido:

```bash
export $(cat .env.example | xargs)
python src/main.py
```

Notas:

- El pipeline usa `DefaultAzureCredential` y funcionará con Managed Identity cuando se ejecute desde un recurso con identidad administrada (VM, Azure Function, etc.). Localmente, puede usar `az login` o un archivo `.env` con credenciales para pruebas.
- Ajusta los nombres de modelos y endpoints según tu despliegue de Foundry.

### Autenticacion

- Habilitar role based access control en la base de datos vectorial
- Habilitar access policy para el id de la cuenta principal en el contenedor blob storage:
  storage>containers>blob>access policy
