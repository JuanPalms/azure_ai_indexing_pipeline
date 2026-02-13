import hashlib
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from .blob_utils import list_blobs, download_blob_to_bytes
from .ocr_foundry import extract_text_from_bytes, get_embeddings_from_foundry, chunk_text
from .search_indexer import (
    create_index_if_not_exists,
    upsert_documents,
    get_indexed_ids,
)
from .config import EMBEDDING_DIM, BATCH_SIZE, MAX_WORKERS, validate_config

log = logging.getLogger("ingestion_pipeline.main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_id(raw: str) -> str:
    """Produce an AI-Search-safe key from an arbitrary string.

    Azure AI Search keys only allow letters, digits, ``-``, ``_``, and ``=``.
    We use a deterministic base16 hash so the id is always valid.
    """
    safe = re.sub(r"[^A-Za-z0-9_=\-]", "_", raw)
    if safe == raw:
        return safe
    # Append a short hash so different originals that collapse to the same
    # sanitized form remain unique.
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{safe}__{short_hash}"


def _batched(iterable: list, size: int):
    """Yield successive slices of *size* from *iterable*."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _build_search_doc(
    blob_name: str,
    chunk_index: int,
    content: str,
    embedding: List[float],
) -> dict:
    raw_id = f"{blob_name}__chunk_{chunk_index}"
    return {
        "@search.action": "upload",
        "id": _sanitize_id(raw_id),
        "content": content,
        "source_blob": blob_name,
        "chunk_index": chunk_index,
        "embedding": embedding,
    }


# ---------------------------------------------------------------------------
# Per-blob processing
# ---------------------------------------------------------------------------

def _process_blob(blob_name: str) -> List[dict]:
    """Download, OCR, chunk, embed, and return search docs for one blob."""
    log.info("Processing: %s", blob_name)

    data = download_blob_to_bytes(blob_name)
    text = extract_text_from_bytes(data)

    if not text or not text.strip():
        log.warning("No text extracted from '%s' — skipping.", blob_name)
        return []

    chunks = chunk_text(text)
    log.info("  '%s' → %d chunk(s)", blob_name, len(chunks))

    docs: List[dict] = []
    for idx, chunk in enumerate(chunks):
        embedding = get_embeddings_from_foundry(chunk)
        if embedding is None:
            log.warning("  Null embedding for chunk %d of '%s' — skipping chunk.", idx, blob_name)
            continue
        if len(embedding) != EMBEDDING_DIM:
            log.warning(
                "  Embedding dim %d ≠ expected %d for chunk %d of '%s'",
                len(embedding), EMBEDDING_DIM, idx, blob_name,
            )
        docs.append(_build_search_doc(blob_name, idx, chunk, embedding))

    return docs


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    # 1. Validate configuration
    validate_config()
    log.info("Configuration validated.")

    # 2. Create index
    log.info("Ensuring search index exists...")
    create_index_if_not_exists()

    # 3. List blobs (already filtered by supported extensions)
    blobs = list_blobs()
    log.info("%d supported blob(s) found in container.", len(blobs))
    if not blobs:
        log.info("Nothing to process — exiting.")
        return

    # 4. Duplicate detection — skip already-indexed blobs
    already_indexed = get_indexed_ids()
    if already_indexed:
        before = len(blobs)
        blobs = [b for b in blobs if b not in already_indexed]
        skipped = before - len(blobs)
        if skipped:
            log.info("Skipping %d already-indexed blob(s).", skipped)
    if not blobs:
        log.info("All blobs already indexed — nothing to do.")
        return

    # 5. Process blobs and upsert per-blob (parallel processing)
    total_indexed = 0
    failed: List[str] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_blob = {
            executor.submit(_process_blob, b): b for b in blobs
        }
        for future in as_completed(future_to_blob):
            blob_name = future_to_blob[future]
            try:
                docs = future.result()
                if not docs:
                    continue
                # Upsert immediately for this blob
                for batch in _batched(docs, BATCH_SIZE):
                    resp = upsert_documents(batch)
                    log.info(
                        "Upserted %d chunk(s) for '%s'. Response: %s",
                        len(batch), blob_name, resp,
                    )
                total_indexed += len(docs)
            except Exception:
                log.exception("Failed to process/index blob '%s'.", blob_name)
                failed.append(blob_name)

    # 6. Summary
    log.info("Total chunks indexed: %d", total_indexed)
    if failed:
        log.warning("The following blobs could not be processed: %s", failed)
    else:
        log.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
