import os
import logging
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

load_dotenv()

# ---------------------------------------------------------------------------
# Managed Identity
# ---------------------------------------------------------------------------
MANAGED_IDENTITY_CLIENT_ID = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")


def get_credential() -> DefaultAzureCredential:
    """Return a ``DefaultAzureCredential`` configured for the pipeline.

    If ``MANAGED_IDENTITY_CLIENT_ID`` is set, it is passed so that Azure
    picks the correct User-assigned Managed Identity.  Otherwise the default
    credential chain is used (works with ``az login`` for local dev).
    """
    kwargs = {}
    if MANAGED_IDENTITY_CLIENT_ID:
        kwargs["managed_identity_client_id"] = MANAGED_IDENTITY_CLIENT_ID
    return DefaultAzureCredential(**kwargs)


# ---------------------------------------------------------------------------
# Azure Blob Storage
# ---------------------------------------------------------------------------
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")
BLOB_CONTAINER_NAME = os.environ.get("BLOB_CONTAINER_NAME")

# ---------------------------------------------------------------------------
# Azure AI Search
# ---------------------------------------------------------------------------
SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.environ.get("SEARCH_INDEX_NAME", "document-index")

# ---------------------------------------------------------------------------
# Microsoft Foundry (embeddings)
# ---------------------------------------------------------------------------
FOUNDRY_ENDPOINT = os.environ.get("FOUNDRY_ENDPOINT")
FOUNDRY_MODEL_ID = os.environ.get("FOUNDRY_MODEL_ID")
FOUNDRY_SCOPE = os.environ.get("FOUNDRY_SCOPE")
FOUNDRY_API_KEY = os.environ.get("FOUNDRY_API_KEY")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))
# "openai" → builds /openai/deployments/{model}/embeddings path
# "custom" → uses FOUNDRY_ENDPOINT as the complete URL
FOUNDRY_API_FORMAT = os.environ.get("FOUNDRY_API_FORMAT", "openai").lower()
FOUNDRY_API_VERSION = os.environ.get("FOUNDRY_API_VERSION", "2024-06-01")

# ---------------------------------------------------------------------------
# Azure Document Intelligence (OCR)
# ---------------------------------------------------------------------------
DOCUMENT_INTELLIGENCE_ENDPOINT = os.environ.get("DOCUMENT_INTELLIGENCE_ENDPOINT")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure and return the pipeline root logger."""
    logger = logging.getLogger("ingestion_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

_REQUIRED = {
    "STORAGE_ACCOUNT_NAME": STORAGE_ACCOUNT_NAME,
    "BLOB_CONTAINER_NAME": BLOB_CONTAINER_NAME,
    "SEARCH_ENDPOINT": SEARCH_ENDPOINT,
    "FOUNDRY_ENDPOINT": FOUNDRY_ENDPOINT,
    "FOUNDRY_MODEL_ID": FOUNDRY_MODEL_ID,
    "DOCUMENT_INTELLIGENCE_ENDPOINT": DOCUMENT_INTELLIGENCE_ENDPOINT,
}


def validate_config():
    """Raise ``ValueError`` if any required env var is missing."""
    missing = [k for k, v in _REQUIRED.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file or environment."
        )
