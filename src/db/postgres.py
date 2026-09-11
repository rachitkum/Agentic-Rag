"""
Postgres: users, documents, and tenant assignment.

Source of truth for user_id -> tenant_id. Every read and write path resolves the tenant
from here before touching Weaviate, which is what lets one heavy user be moved to a
dedicated tenant without resharding everyone.

Both tables are hash-partitioned on user_id, so each partition keeps its own smaller
indexes and a lookup for one user touches only one partition. documents is partitioned
on user_id (not doc_id) to keep a user's rows co-located with their users row.

Env:
    POSTGRES_URL     (e.g. postgresql://rag:rag@localhost:5432/rag)
    PG_PARTITIONS    (default 64; fixed once the tables exist)
"""

import os
import time
from contextlib import contextmanager

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag")

# Partition count is fixed at creation time: changing it means rebuilding the tables,
# so it is deliberately over-provisioned relative to current load.
PG_PARTITIONS = int(os.getenv("PG_PARTITIONS", "64"))

# Opened lazily so importing this module never forces a DB connection.
_POOL: ConnectionPool | None = None

# Cached user count; a full count across every partition is too slow per signup.
_USER_COUNT: int | None = None
_USER_COUNT_AT: float = 0.0
_USER_COUNT_TTL = 60.0


def get_pool() -> ConnectionPool:
    """Return the process-wide connection pool, opening it on first use."""
    global _POOL
    if _POOL is None:
        _POOL = ConnectionPool(POSTGRES_URL, min_size=2, max_size=20, open=True)
    return _POOL


@contextmanager
def get_cursor(commit: bool = False):
    """Borrow a connection from the pool and yield a dict-returning cursor."""
    with get_pool().connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
        if commit:
            conn.commit()


# Partitioned tables need the partition key in every unique constraint, hence the
# composite primary key on documents.
_PARENT_TABLES = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'ACTIVE',   -- ACTIVE | MIGRATING
    doc_count   INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id)
) PARTITION BY HASH (user_id);

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT        NOT NULL,
    user_id     TEXT        NOT NULL,
    session_id  TEXT,
    file_name   TEXT        NOT NULL DEFAULT '',
    status      TEXT        NOT NULL DEFAULT 'PROCESSING',  -- PROCESSING | READY | FAILED
    node_count  INTEGER     NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, user_id)
) PARTITION BY HASH (user_id);
"""

# idx_users_tenant serves the rebalancing job's scan by bucket.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_users_tenant     ON users (tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_user   ON documents (user_id);
"""


def _partition_ddl() -> str:
    """Emit CREATE TABLE ... PARTITION OF for every hash partition."""
    stmts = []
    for i in range(PG_PARTITIONS):
        for parent in ("users", "documents"):
            stmts.append(
                f"CREATE TABLE IF NOT EXISTS {parent}_p{i:03d} "
                f"PARTITION OF {parent} "
                f"FOR VALUES WITH (MODULUS {PG_PARTITIONS}, REMAINDER {i});"
            )
    return "\n".join(stmts)


def initSchema() -> None:
    """Create the partitioned tables and their partitions. Safe to re-run on boot."""
    with get_cursor(commit=True) as cur:
        cur.execute(_PARENT_TABLES)
        cur.execute(_partition_ddl())
        cur.execute(_INDEXES)


# --- users ----------------------------------------------------------------

def insertUser(user_id: str, tenant_id: str) -> dict:
    """Insert a user, or return the existing row if they're already registered."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO users (user_id, tenant_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO NOTHING
            RETURNING user_id, tenant_id, status, doc_count, created_at
            """,
            (user_id, tenant_id),
        )
        row = cur.fetchone()

    return row if row is not None else getUserRow(user_id)


def getUserCount() -> int:
    """Live user count, cached. Feeds the bucket count for new signups.

    Only ever grows: a dip (deletions) must not narrow the range new users hash into.
    """
    global _USER_COUNT, _USER_COUNT_AT
    now = time.monotonic()
    if _USER_COUNT is None or now - _USER_COUNT_AT > _USER_COUNT_TTL:
        with get_cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM users")
            fresh = cur.fetchone()["n"]
        _USER_COUNT = max(fresh, _USER_COUNT or 0)
        _USER_COUNT_AT = now
    return _USER_COUNT


def getUserRow(user_id: str) -> dict | None:
    """Fetch a user row, or None if they don't exist."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT user_id, tenant_id, status, doc_count, created_at
            FROM users WHERE user_id = %s
            """,
            (user_id,),
        )
        return cur.fetchone()


def updateUserTenant(user_id: str, tenant_id: str) -> None:
    """Point a user at a different tenant (the cutover step of a migration)."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE users SET tenant_id = %s WHERE user_id = %s",
            (tenant_id, user_id),
        )


def setUserStatus(user_id: str, status: str) -> None:
    """Set ACTIVE or MIGRATING. Uploads are queued while a user is MIGRATING."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE users SET status = %s WHERE user_id = %s",
            (status, user_id),
        )


def incrementDocCount(user_id: str, delta: int = 1) -> None:
    """Adjust a user's document count; feeds the whale-promotion trigger."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE users SET doc_count = doc_count + %s WHERE user_id = %s",
            (delta, user_id),
        )


def listUsersByTenant(tenant_id: str, limit: int = 100) -> list[dict]:
    """Heaviest users in one tenant -- the promotion job's candidate list."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT user_id, tenant_id, doc_count
            FROM users WHERE tenant_id = %s
            ORDER BY doc_count DESC
            LIMIT %s
            """,
            (tenant_id, limit),
        )
        return cur.fetchall()


# --- documents ------------------------------------------------------------

def insertDocument(doc_id: str, user_id: str, file_name: str, session_id: str = "") -> None:
    """Record a document at upload time, before ingestion runs."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO documents (doc_id, user_id, session_id, file_name, status)
            VALUES (%s, %s, %s, %s, 'PROCESSING')
            ON CONFLICT (doc_id, user_id) DO NOTHING
            """,
            (doc_id, user_id, session_id, file_name),
        )


def setDocumentStatus(doc_id: str, user_id: str, status: str, node_count: int = 0) -> None:
    """Mark a document READY or FAILED once its ingestion job finishes."""
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE documents SET status = %s, node_count = %s
            WHERE doc_id = %s AND user_id = %s
            """,
            (status, node_count, doc_id, user_id),
        )


def getDocumentsByUser(user_id: str) -> list[dict]:
    """All of a user's documents. Hits a single partition."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, user_id, session_id, file_name, status, node_count, created_at
            FROM documents WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return cur.fetchall()
