import logging
import time
from typing import Dict, List, Tuple

import requests
from azure.ai.formrecognizer import DocumentAnalysisClient

from .config import (
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    FOUNDRY_ENDPOINT,
    FOUNDRY_MODEL_ID,
    FOUNDRY_SCOPE,
    FOUNDRY_API_KEY,
    FOUNDRY_API_FORMAT,
    FOUNDRY_API_VERSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_credential,
)

log = logging.getLogger("ingestion_pipeline.ocr_foundry")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds


def _retry(fn, *args, retries: int = _MAX_RETRIES, **kwargs):
    """Execute *fn* with exponential-backoff retries on transient errors."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = _BACKOFF_BASE ** attempt
                log.warning(
                    "Attempt %d/%d failed (%s). Retrying in %ds...",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error("All %d attempts failed.", retries)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OCR — Document Intelligence (prebuilt-layout for structural metadata)
# ---------------------------------------------------------------------------

def extract_document(doc_bytes: bytes) -> Dict:
    """Extract structured text from a document using Azure Document Intelligence.

    Uses ``prebuilt-layout`` to obtain paragraph-level metadata including
    page numbers and structural roles (title, sectionHeading, etc.).

    Returns a dict::

        {
            "paragraphs": [{"text": str, "page": int, "section": str}, ...],
            "total_pages": int,
        }
    """
    if not DOCUMENT_INTELLIGENCE_ENDPOINT:
        raise ValueError("DOCUMENT_INTELLIGENCE_ENDPOINT not configured")

    credential = get_credential()
    client = DocumentAnalysisClient(
        endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT, credential=credential,
    )

    def _analyze():
        poller = client.begin_analyze_document("prebuilt-layout", doc_bytes)
        return poller.result()

    result = _retry(_analyze)

    total_pages = len(result.pages) if result.pages else 1

    paragraphs: List[Dict] = []
    current_section = ""

    for para in (result.paragraphs or []):
        page = 1
        if para.bounding_regions:
            page = para.bounding_regions[0].page_number

        # Track section headings
        if para.role in ("title", "sectionHeading"):
            current_section = para.content

        paragraphs.append({
            "text": para.content,
            "page": page,
            "section": current_section,
        })

    # Fallback: if no paragraphs, iterate pages/lines
    if not paragraphs:
        for page in (result.pages or []):
            for line in (page.lines or []):
                paragraphs.append({
                    "text": line.content,
                    "page": page.page_number,
                    "section": "",
                })

    log.info(
        "Extracted %d paragraph(s) across %d page(s).",
        len(paragraphs), total_pages,
    )
    return {"paragraphs": paragraphs, "total_pages": total_pages}


# ---------------------------------------------------------------------------
# Metadata-aware chunking
# ---------------------------------------------------------------------------

def chunk_paragraphs(
    paragraphs: List[Dict],
    total_pages: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Split annotated paragraphs into chunks preserving page/section metadata.

    Each returned chunk is a dict::

        {
            "text": str,
            "page_start": int,
            "page_end": int,
            "section": str,
            "total_pages": int,
        }
    """
    if not paragraphs:
        return []

    # Build word-level annotations: (word, page, section)
    annotated_words: List[Tuple[str, int, str]] = []
    for para in paragraphs:
        for w in para["text"].split():
            annotated_words.append((w, para["page"], para["section"]))

    if not annotated_words:
        return []

    chunks: List[Dict] = []
    current: List[Tuple[str, int, str]] = []
    current_len = 0

    for aw in annotated_words:
        word = aw[0]
        word_len = len(word) + (1 if current else 0)

        if current_len + word_len > chunk_size and current:
            # Emit chunk
            text = " ".join(w for w, _, _ in current)
            pages = [p for _, p, _ in current]
            chunks.append({
                "text": text,
                "page_start": min(pages),
                "page_end": max(pages),
                "section": current[0][2],  # section of first word
                "total_pages": total_pages,
            })
            # Keep overlap words
            overlap_words: List[Tuple[str, int, str]] = []
            overlap_len = 0
            for wt in reversed(current):
                if overlap_len + len(wt[0]) + 1 > overlap:
                    break
                overlap_words.insert(0, wt)
                overlap_len += len(wt[0]) + 1
            current = overlap_words
            current_len = sum(len(w) for w, _, _ in current) + max(len(current) - 1, 0)

        current.append(aw)
        current_len += word_len

    # Last chunk
    if current:
        text = " ".join(w for w, _, _ in current)
        pages = [p for _, p, _ in current]
        chunks.append({
            "text": text,
            "page_start": min(pages),
            "page_end": max(pages),
            "section": current[0][2],
            "total_pages": total_pages,
        })

    return chunks


# ---------------------------------------------------------------------------
# Embeddings — Microsoft Foundry
# ---------------------------------------------------------------------------

def _get_bearer_for_foundry() -> str | None:
    """Get an authentication token/key for the Foundry endpoint."""
    if FOUNDRY_API_KEY:
        return FOUNDRY_API_KEY
    if FOUNDRY_SCOPE:
        cred = get_credential()
        token = cred.get_token(FOUNDRY_SCOPE)
        return token.token
    return None


def _build_embeddings_url() -> str:
    """Build the embeddings endpoint URL based on ``FOUNDRY_API_FORMAT``.

    Supported formats:

    - ``"openai"`` (default): Azure OpenAI-compatible path::

          {base}/openai/deployments/{FOUNDRY_MODEL_ID}/embeddings?api-version=...

    - ``"custom"``: ``FOUNDRY_ENDPOINT`` is used as the complete URL.
      Useful for non-OpenAI models or custom inference servers.
    """
    base = FOUNDRY_ENDPOINT.rstrip("/")

    if FOUNDRY_API_FORMAT == "custom":
        return base

    # Default: OpenAI-compatible
    return (
        f"{base}/openai/deployments/{FOUNDRY_MODEL_ID}"
        f"/embeddings?api-version={FOUNDRY_API_VERSION}"
    )


def _build_embeddings_payload(text: str) -> dict:
    """Build the request body based on ``FOUNDRY_API_FORMAT``.

    - ``"openai"``: ``{"input": text}`` (model is in the URL path).
    - ``"custom"``: ``{"model": ..., "input": text}`` (model in body).
    """
    if FOUNDRY_API_FORMAT == "custom":
        return {"model": FOUNDRY_MODEL_ID, "input": text}
    return {"input": text}


def get_embeddings_from_foundry(text: str) -> List[float]:
    """Call the Foundry inference endpoint to obtain embeddings for *text*.

    Supports ``FOUNDRY_API_FORMAT`` = ``"openai"`` | ``"custom"``.
    Handles multiple response shapes (``data[0].embedding``, ``embedding``,
    raw array).
    """
    if not FOUNDRY_ENDPOINT or not FOUNDRY_MODEL_ID:
        raise ValueError("FOUNDRY_ENDPOINT and FOUNDRY_MODEL_ID must be configured")

    url = _build_embeddings_url()
    payload = _build_embeddings_payload(text)

    bearer = _get_bearer_for_foundry()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if FOUNDRY_API_KEY:
        headers["api-key"] = FOUNDRY_API_KEY
    elif bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    def _post():
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if not resp.ok:
            log.error(
                "Foundry returned HTTP %d: %s", resp.status_code,
                resp.text[:500] if resp.text else "(empty body)",
            )
        resp.raise_for_status()
        return resp

    resp = _retry(_post)

    if not resp.text:
        raise ValueError(f"Empty response from Foundry (HTTP {resp.status_code})")

    data = resp.json()
    log.debug("Foundry response keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))

    # OpenAI format: {"data": [{"embedding": [...]}]}
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        first = data["data"][0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
    # Direct embedding field
    if isinstance(data, dict) and "embedding" in data:
        return data["embedding"]
    # Raw array
    if isinstance(data, list):
        return data[0]

    raise ValueError(
        f"Could not parse embedding from Foundry response: "
        f"{list(data.keys()) if isinstance(data, dict) else data}"
    )
