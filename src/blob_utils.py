import io
import logging
import os
from typing import List

from azure.storage.blob import BlobServiceClient

from .config import STORAGE_ACCOUNT_NAME, BLOB_CONTAINER_NAME, SUPPORTED_EXTENSIONS, get_credential

log = logging.getLogger("ingestion_pipeline.blob_utils")


def _get_blob_service_client() -> BlobServiceClient:
    if not STORAGE_ACCOUNT_NAME:
        raise ValueError("STORAGE_ACCOUNT_NAME is not configured")
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    credential = get_credential()
    return BlobServiceClient(account_url=account_url, credential=credential)


def _is_supported(blob_name: str) -> bool:
    """Return True if the blob has a supported file extension."""
    ext = os.path.splitext(blob_name)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def list_blobs(prefix: str = "", filter_supported: bool = True) -> List[str]:
    """List blobs in the configured container.

    When *filter_supported* is True (default) only blobs with extensions in
    ``SUPPORTED_EXTENSIONS`` (.pdf, .docx, .doc) are returned.
    """
    svc = _get_blob_service_client()
    container = svc.get_container_client(BLOB_CONTAINER_NAME)
    all_blobs = [b.name for b in container.list_blobs(name_starts_with=prefix)]

    if filter_supported:
        filtered = [b for b in all_blobs if _is_supported(b)]
        skipped = len(all_blobs) - len(filtered)
        if skipped:
            log.info(
                "Filtered out %d blob(s) with unsupported extensions", skipped,
            )
        return filtered

    return all_blobs


def download_blob_to_bytes(blob_name: str) -> bytes:
    """Download a blob to memory and return its raw bytes."""
    svc = _get_blob_service_client()
    container = svc.get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container.get_blob_client(blob_name)
    stream = io.BytesIO()
    download_stream = blob_client.download_blob()
    download_stream.readinto(stream)
    log.debug("Downloaded blob '%s' (%d bytes)", blob_name, stream.tell())
    return stream.getvalue()
