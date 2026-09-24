"""
Out-of-process PDF ingestion.

The API stores the upload in S3 and enqueues a task here, so a RAPTOR build no longer
competes with chat requests for the API's threadpool, survives an API restart, and
retries on failure. Job status stays in Valkey (src/db/memory.py) -- the client polls
that, so Celery's result backend is unused.

Its own module because `celery -A src.worker worker` imports it directly: pointing
Celery at src.server.app would boot the whole Starlette app (routes, CORS, the
Postgres pool in on_startup) inside every worker, and import circularly besides.

Run a worker:
    celery -A src.worker worker --loglevel=info
"""

import os

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from dotenv import load_dotenv

from src.db import memory
from src.server.ingest import ingest_from_s3

load_dotenv()

BROKER_URL = os.getenv("CELERY_BROKER_URL") or os.getenv(
    "VALKEY_URL", "redis://localhost:6379/0"
)

# Each concurrent ingest holds a document's embeddings in memory and fits
# GaussianMixture repeatedly, so this is the real memory knob.
INGEST_CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "2"))

app = Celery("ingest", broker=BROKER_URL)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # Ack AFTER the task finishes, not when it is picked up. A worker killed mid-ingest
    # (deploy, OOM) returns the task to the queue instead of dropping it -- the whole
    # reason for moving ingestion out of the API process.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Fetch one task at a time: the default prefetch grabs a batch, which with long
    # ingests leaves work queued behind a busy worker while others sit idle.
    worker_prefetch_multiplier=1,
    worker_concurrency=INGEST_CONCURRENCY,

    # Redelivery window. Must exceed the slowest realistic ingest or the broker hands
    # the task to a second worker while the first is still building the tree.
    broker_transport_options={"visibility_timeout": 60 * 60},

    # Ceiling so a pathological PDF can't hold a worker slot forever. The soft limit
    # raises inside the task first, leaving room to record the failure.
    task_time_limit=60 * 30,
    task_soft_time_limit=60 * 25,
)


@app.task(
    name="ingest.pdf",
    bind=True,
    max_retries=3,
    # Exponential backoff: a failed ingest is usually Weaviate or the LLM being
    # briefly unavailable, and retrying immediately just fails again.
    default_retry_delay=60,
    retry_backoff=True,
    retry_jitter=True,
)
def ingest_pdf_task(self, job_id: str, s3_key: str, user_id: str, tenant_id: str,
                    doc_id: str, session_id: str = "", file_name: str = "") -> dict:
    """Fetch the upload from S3, build its RAPTOR tree, record the outcome in Valkey.

    Safe to run twice: ingest_from_s3 clears any existing nodes for this doc_id before
    inserting, so a redelivered task replaces rather than duplicates them.
    """
    try:
        result = ingest_from_s3(s3_key, user_id, tenant_id, doc_id, session_id, file_name)
    except SoftTimeLimitExceeded:
        # Out of time rather than broken; retrying the same PDF would time out again.
        print(f"INGEST timed out: job={job_id} key={s3_key}")
        _recordError(job_id, "Ingestion timed out")
        raise
    except Exception as e:
        print(f"ERROR during ingestion: job={job_id} key={s3_key}: {e}")
        if self.request.retries >= self.max_retries:
            # Out of attempts -- this is the client's only signal that it failed.
            _recordError(job_id, str(e))
            raise
        # Leave the job "processing" between attempts; the S3 object is still there.
        raise self.retry(exc=e)

    try:
        memory.setJobDone(job_id, result)
    except Exception as e:
        # The work landed in Weaviate; only the status write failed. Retrying would
        # re-ingest, so record the loss and let the client's poll expire.
        print(f"ERROR recording job completion: job={job_id}: {e}")
    return result


def _recordError(job_id: str, error: str) -> None:
    try:
        memory.setJobError(job_id, error)
    except Exception as e:
        print(f"ERROR recording job failure: job={job_id}: {e}")
