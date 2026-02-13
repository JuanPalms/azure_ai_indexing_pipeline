import json
import logging
import time
from typing import List, Set

import requests

from .config import SEARCH_ENDPOINT, SEARCH_INDEX_NAME, EMBEDDING_DIM, get_credential

log = logging.getLogger("ingestion_pipeline.search_indexer")

API_VERSION = "2025-09-01"

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE = 2


def _retry_request(method: str, url: str, headers: dict, **kwargs):
    """Execute an HTTP request with exponential-backoff retries."""
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE ** attempt
                log.warning(
                    "Request %s %s attempt %d/%d failed (%s). Retrying in %ds...",
                    method, url, attempt, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error("All %d attempts for %s %s failed.", _MAX_RETRIES, method, url)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _get_search_bearer() -> str:
    cred = get_credential()
    token = cred.get_token("https://search.azure.com/.default")
    return token.token


def _auth_headers() -> dict:
    """Return common headers with bearer token."""
    return {
        "Authorization": f"Bearer {_get_search_bearer()}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def index_exists() -> bool:
    """Check whether the search index already exists."""
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX_NAME}?api-version={API_VERSION}"
    headers = _auth_headers()
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def create_index_if_not_exists():
    """Create the vector search index if it does not exist yet.

    Uses the 2024-07-01 schema with ``vectorSearch`` profiles and an HNSW
    algorithm configuration.
    """
    if index_exists():
        log.info("Index '%s' already exists — skipping creation.", SEARCH_INDEX_NAME)
        return

    url = f"{SEARCH_ENDPOINT}/indexes?api-version={API_VERSION}"
    body = {
        "name": SEARCH_INDEX_NAME,
        "fields": [
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
            },
            {
                "name": "source_blob",
                "type": "Edm.String",
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "chunk_index",
                "type": "Edm.Int32",
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "page_start",
                "type": "Edm.Int32",
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "page_end",
                "type": "Edm.Int32",
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "section_heading",
                "type": "Edm.String",
                "filterable": True,
                "searchable": True,
            },
            {
                "name": "total_pages",
                "type": "Edm.Int32",
                "filterable": True,
                "searchable": False,
            },
            {
                "name": "embedding",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": EMBEDDING_DIM,
                "vectorSearchProfile": "default-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "hnsw-algo",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                }
            ],
            "profiles": [
                {
                    "name": "default-profile",
                    "algorithm": "hnsw-algo",
                }
            ],
        },
    }

    _retry_request("POST", url, _auth_headers(), json=body)
    log.info("Index '%s' created successfully.", SEARCH_INDEX_NAME)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def get_indexed_ids(field: str = "source_blob") -> Set[str]:
    """Return all distinct *source_blob* values currently in the index.

    Uses a simple ``search=*&$select=source_blob`` call. For very large
    indices a scroll/continuation approach would be needed.
    """
    url = (
        f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX_NAME}/docs"
        f"?api-version={API_VERSION}&search=*&$select={field}&$top=100000"
    )
    headers = _auth_headers()
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            log.warning("Could not fetch indexed docs for dedup (HTTP %d)", resp.status_code)
            return set()
        data = resp.json()
        return {doc.get(field, "") for doc in data.get("value", [])}
    except requests.RequestException as exc:
        log.warning("Dedup query failed: %s — proceeding without dedup.", exc)
        return set()


# ---------------------------------------------------------------------------
# Document upsert
# ---------------------------------------------------------------------------

def upsert_documents(docs: List[dict]) -> dict:
    """Upsert documents into the search index.

    Each doc must include: id, content, source_blob, chunk_index, embedding.
    """
    url = (
        f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX_NAME}"
        f"/docs/index?api-version={API_VERSION}"
    )
    payload = {"value": docs}
    resp = _retry_request("POST", url, _auth_headers(), json=payload)
    return resp.json()
