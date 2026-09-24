"""
Object storage for uploaded PDFs.

The API writes the raw bytes here and hands the worker a key, so the file never has
to travel through the queue and never sits in the API process's memory for the length
of an ingest.

Env:
    S3_BUCKET             bucket name (required)
    AWS_REGION            default us-east-1
    AWS_ACCESS_KEY_ID     standard boto3 credentials; omit to use the instance role
    AWS_SECRET_ACCESS_KEY
    S3_ENDPOINT_URL       set for MinIO or another S3-compatible server; unset for AWS
"""

import os
import threading

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Set for MinIO ("http://minio:9000"); leave unset to talk to real AWS S3.
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL") or None

_CLIENT = None

# Same double-checked pattern as the Weaviate and Valkey clients: uploads and the
# worker both reach this from threadpool workers.
_CLIENT_LOCK = threading.Lock()


def get_client():
    """Return the shared S3 client, connecting on first use."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    with _CLIENT_LOCK:
        if _CLIENT is None:
            # Retries cover the transient 5xx/timeout that S3 is expected to return
            # occasionally; without this a blip fails an otherwise healthy upload.
            _CLIENT = boto3.client(
                "s3",
                region_name=AWS_REGION,
                endpoint_url=S3_ENDPOINT_URL,
                config=Config(retries={"max_attempts": 3, "mode": "standard"}),
            )
        return _CLIENT


def pdf_key(user_id: str, doc_id: str) -> str:
    """Key for one upload. doc_id is a uuid, so this is unique per document."""
    return f"uploads/{user_id}/{doc_id}.pdf"


def put_pdf(key: str, data: bytes) -> None:
    """Store the raw upload. Raises on failure."""
    get_client().put_object(
        Bucket=S3_BUCKET, Key=key, Body=data, ContentType="application/pdf"
    )


def get_pdf(key: str) -> bytes:
    """Fetch an upload back. Raises if the key is missing."""
    obj = get_client().get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def delete_pdf(key: str) -> None:
    """Best-effort cleanup; a leftover object is harmless, a failed delete is not."""
    try:
        get_client().delete_object(Bucket=S3_BUCKET, Key=key)
    except Exception as e:
        print("ERROR deleting s3 object:", key, e)
