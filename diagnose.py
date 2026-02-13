"""Quick diagnostic — run from the project root:
    python diagnose.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("1. Environment variables")
print("=" * 60)
for var in [
    "MANAGED_IDENTITY_CLIENT_ID",
    "STORAGE_ACCOUNT_NAME",
    "BLOB_CONTAINER_NAME",
]:
    val = os.environ.get(var)
    print(f"  {var} = {val!r}")

mi_client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
storage_account = os.environ.get("STORAGE_ACCOUNT_NAME")
container_name = os.environ.get("BLOB_CONTAINER_NAME")

print()
print("=" * 60)
print("2. Credential chain (with DEBUG logging)")
print("=" * 60)

import logging
logging.basicConfig(level=logging.DEBUG)
azure_logger = logging.getLogger("azure.identity")
azure_logger.setLevel(logging.DEBUG)

from azure.identity import DefaultAzureCredential

kwargs = {}
if mi_client_id:
    # Strip quotes just in case dotenv left them
    mi_client_id = mi_client_id.strip('"').strip("'")
    kwargs["managed_identity_client_id"] = mi_client_id
    print(f"  Using managed_identity_client_id = {mi_client_id}")
else:
    print("  No MANAGED_IDENTITY_CLIENT_ID set — using default chain")

try:
    cred = DefaultAzureCredential(**kwargs)
    # Request a token for Blob Storage
    token = cred.get_token("https://storage.azure.com/.default")
    print(f"\n  SUCCESS — got token (expires: {token.expires_on})")
except Exception as e:
    print(f"\n  FAILED — {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("3. Blob Storage — list blobs")
print("=" * 60)

from azure.storage.blob import BlobServiceClient

try:
    account_url = f"https://{storage_account}.blob.core.windows.net"
    cred2 = DefaultAzureCredential(**kwargs)
    svc = BlobServiceClient(account_url=account_url, credential=cred2)
    container = svc.get_container_client(container_name)
    blobs = [b.name for b in container.list_blobs()]
    print(f"  SUCCESS — {len(blobs)} blob(s) found:")
    for b in blobs[:10]:
        print(f"    - {b}")
    if len(blobs) > 10:
        print(f"    ... and {len(blobs) - 10} more")
except Exception as e:
    print(f"  FAILED — {type(e).__name__}: {e}")
