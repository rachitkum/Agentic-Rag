"""
Postgres: users, tenant assignment, and long-term chat memory.

Source of truth for user_id -> tenant_id, and the durable store for chat turns.
Both tables are hash-partitioned on user_id; chat_messages is range-partitioned by
month underneath so old months can be detached and archived.

Env:
    POSTGRES_URL      (e.g. postgresql://rag:rag@localhost:5432/rag)
    PG_PARTITIONS     (default 64; fixed once the tables exist)
    PG_MONTHS_AHEAD   (default 2; month partitions kept ahead of now)
"""

import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag")

# Fixed at creation: changing it means rebuilding the tables.
PG_PARTITIONS = int(os.getenv("PG_PARTITIONS", "64"))

MONTHS_AHEAD = int(os.getenv("PG_MONTHS_AHEAD", "2"))

# Lazy so importing this module never opens a connection.
_POOL: ConnectionPool | None = None

# Cached: a full count per signup is too slow.
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


# Partition key must be in every unique constraint, hence created_at in the PK.
_PARENT_TABLES = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id)
) PARTITION BY HASH (user_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    user_id     TEXT        NOT NULL,
    session_id  TEXT        NOT NULL,
    seq         BIGINT      NOT NULL,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, session_id, seq, created_at)
) PARTITION BY HASH (user_id);
"""

# chat_messages is read by its primary key, so it needs no extra index.
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users (tenant_id);
"""


def _hash_partition_ddl() -> str:
    """Hash partitions for both tables; chat_messages ones sub-partition by month."""
    stmts = []
    for i in range(PG_PARTITIONS):
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS users_p{i:03d} PARTITION OF users "
            f"FOR VALUES WITH (MODULUS {PG_PARTITIONS}, REMAINDER {i});"
        )
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS chat_messages_p{i:03d} PARTITION OF chat_messages "
            f"FOR VALUES WITH (MODULUS {PG_PARTITIONS}, REMAINDER {i}) "
            f"PARTITION BY RANGE (created_at);"
        )
    return "\n".join(stmts)


def _month_bounds(when: datetime) -> tuple[str, str, str]:
    """(suffix, start, end) for the month containing `when`."""
    start = datetime(when.year, when.month, 1, tzinfo=timezone.utc)
    end = datetime(when.year + (when.month == 12), when.month % 12 + 1, 1, tzinfo=timezone.utc)
    return start.strftime("%Y_%m"), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _month_partition_ddl(months_ahead: int = MONTHS_AHEAD) -> str:
    """Month partitions for now + months_ahead, so a write never hits a missing range."""
    stmts = []
    now = datetime.now(timezone.utc)
    for m in range(months_ahead + 1):
        suffix, start, end = _month_bounds(now + relativedelta(months=m))
        for i in range(PG_PARTITIONS):
            stmts.append(
                f"CREATE TABLE IF NOT EXISTS chat_messages_p{i:03d}_{suffix} "
                f"PARTITION OF chat_messages_p{i:03d} "
                f"FOR VALUES FROM ('{start}') TO ('{end}');"
            )
    return "\n".join(stmts)


def ensureMonthPartitions(months_ahead: int = MONTHS_AHEAD) -> None:
    """Create upcoming month partitions. Run on boot and from a monthly job."""
    with get_cursor(commit=True) as cur:
        cur.execute(_month_partition_ddl(months_ahead))


def initSchema() -> None:
    """Create the partitioned tables and their partitions. Safe to re-run on boot."""
    with get_cursor(commit=True) as cur:
        cur.execute(_PARENT_TABLES)
        cur.execute(_hash_partition_ddl())
        cur.execute(_month_partition_ddl())
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
            RETURNING user_id, tenant_id, created_at
            """,
            (user_id, tenant_id),
        )
        row = cur.fetchone()

    return row if row is not None else getUserRow(user_id)


def getUserCount() -> int:
    """Live user count, cached. Only ever grows, so the bucket range never narrows."""
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
            SELECT user_id, tenant_id, created_at
            FROM users WHERE user_id = %s
            """,
            (user_id,),
        )
        return cur.fetchone()


# --- chat messages --------------------------------------------------------

def appendMessages(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Append turns to the durable log, numbering them after the current last seq."""
    if not turns:
        return
    with get_cursor(commit=True) as cur:
        cur.execute(
            """
            SELECT coalesce(max(seq), 0) AS last FROM chat_messages
            WHERE user_id = %s AND session_id = %s
            """,
            (user_id, session_id),
        )
        seq = cur.fetchone()["last"]
        cur.executemany(
            """
            INSERT INTO chat_messages (user_id, session_id, seq, role, content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, session_id, seq) DO NOTHING
            """,
            [
                (user_id, session_id, seq + i, t["role"], t["content"])
                for i, t in enumerate(turns, start=1)
            ],
        )


def getRecentMessages(user_id: str, session_id: str, limit: int) -> list[dict]:
    """Last `limit` turns, oldest first. Used to warm the cache on a miss."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT role, content FROM chat_messages
            WHERE user_id = %s AND session_id = %s
            ORDER BY seq DESC LIMIT %s
            """,
            (user_id, session_id, limit),
        )
        return list(reversed(cur.fetchall()))


def getMessagesBefore(user_id: str, session_id: str, before_seq: int | None,
                      limit: int = 50) -> list[dict]:
    """One page of older turns, newest first. Cursor-paginated on seq, never OFFSET."""
    with get_cursor() as cur:
        if before_seq is None:
            cur.execute(
                """
                SELECT seq, role, content FROM chat_messages
                WHERE user_id = %s AND session_id = %s
                ORDER BY seq DESC LIMIT %s
                """,
                (user_id, session_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT seq, role, content FROM chat_messages
                WHERE user_id = %s AND session_id = %s AND seq < %s
                ORDER BY seq DESC LIMIT %s
                """,
                (user_id, session_id, before_seq, limit),
            )
        return cur.fetchall()


# --- partition maintenance ------------------------------------------------

def listMonthPartitions() -> list[dict]:
    """Every chat_messages month partition, oldest first."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT c.relname AS name,
                   pg_total_relation_size(c.oid) AS bytes
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = current_schema()
              AND c.relname ~ '^chat_messages_p[0-9]{3}_[0-9]{4}_[0-9]{2}$'
            ORDER BY right(c.relname, 7), c.relname
            """
        )
        return cur.fetchall()


def detachMonthPartitions(before: datetime) -> list[str]:
    """Detach month partitions older than `before`. They survive as standalone tables:
    export, then drop."""
    cutoff = before.strftime("%Y_%m")
    detached = []
    with get_cursor(commit=True) as cur:
        for row in listMonthPartitions():
            name = row["name"]
            if name[-7:] >= cutoff:
                continue
            parent = name[:name.rindex("_", 0, name.rindex("_"))]
            cur.execute(f"ALTER TABLE {parent} DETACH PARTITION {name}")
            detached.append(name)
    return detached
