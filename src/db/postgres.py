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
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://rag:rag@localhost:5432/rag")

# Fixed at creation: changing it means rebuilding the tables.
PG_PARTITIONS = int(os.getenv("PG_PARTITIONS", "64"))

MONTHS_AHEAD = int(os.getenv("PG_MONTHS_AHEAD", "2"))

# Per process. N replicas x POOL_MAX must stay under the server's max_connections.
POOL_MAX = int(os.getenv("PG_POOL_MAX", "10"))

# Lazy so importing this module never opens a connection.
_POOL: AsyncConnectionPool | None = None

# Cached: a full count per signup is too slow.
_USER_COUNT: int | None = None
_USER_COUNT_AT: float = 0.0
_USER_COUNT_TTL = 60.0


async def get_pool() -> AsyncConnectionPool:
    """Return the process-wide connection pool, opening it on first use.

    max_size is per process: with N replicas the cluster holds N x max_size
    connections, so keep it well under the server's max_connections or front it
    with PgBouncer.
    """
    global _POOL
    if _POOL is None:
        _POOL = AsyncConnectionPool(POSTGRES_URL, min_size=1, max_size=POOL_MAX,
                                    open=False)
        await _POOL.open()
    return _POOL


async def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        await _POOL.close()
        _POOL = None


@asynccontextmanager
async def get_cursor(commit: bool = False):
    """Borrow a connection from the pool and yield a dict-returning cursor."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            yield cur
        if commit:
            await conn.commit()


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


async def ensureMonthPartitions(months_ahead: int = MONTHS_AHEAD) -> None:
    """Create upcoming month partitions. Run on boot and from a monthly job."""
    async with get_cursor(commit=True) as cur:
        await cur.execute(_month_partition_ddl(months_ahead))


async def initSchema() -> None:
    """Create the partitioned tables and their partitions. Safe to re-run on boot."""
    async with get_cursor(commit=True) as cur:
        await cur.execute(_PARENT_TABLES)
        await cur.execute(_hash_partition_ddl())
        await cur.execute(_month_partition_ddl())
        await cur.execute(_INDEXES)


# --- users ----------------------------------------------------------------

async def insertUser(user_id: str, tenant_id: str) -> dict:
    """Insert a user, or return the existing row if they're already registered."""
    async with get_cursor(commit=True) as cur:
        await cur.execute(
            """
            INSERT INTO users (user_id, tenant_id)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO NOTHING
            RETURNING user_id, tenant_id, created_at
            """,
            (user_id, tenant_id),
        )
        row = await cur.fetchone()

    return row if row is not None else await getUserRow(user_id)


async def getUserCount() -> int:
    """Live user count, cached. Only ever grows, so the bucket range never narrows."""
    global _USER_COUNT, _USER_COUNT_AT
    now = time.monotonic()
    if _USER_COUNT is None or now - _USER_COUNT_AT > _USER_COUNT_TTL:
        async with get_cursor() as cur:
            await cur.execute("SELECT count(*) AS n FROM users")
            fresh = (await cur.fetchone())["n"]
        _USER_COUNT = max(fresh, _USER_COUNT or 0)
        _USER_COUNT_AT = now
    return _USER_COUNT


async def getUserRow(user_id: str) -> dict | None:
    """Fetch a user row, or None if they don't exist."""
    async with get_cursor() as cur:
        await cur.execute(
            """
            SELECT user_id, tenant_id, created_at
            FROM users WHERE user_id = %s
            """,
            (user_id,),
        )
        return await cur.fetchone()


# --- chat messages --------------------------------------------------------

async def appendMessages(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Append turns to the durable log.

    seq is computed inside the INSERT so concurrent writers can't read the same max
    and collide; the unique index on (user_id, session_id, seq) is the backstop.
    """
    if not turns:
        return
    rows = [(t["role"], t["content"]) for t in turns]
    async with get_cursor(commit=True) as cur:
        await cur.execute(
            """
            INSERT INTO chat_messages (user_id, session_id, seq, role, content)
            SELECT %s, %s,
                   coalesce((SELECT max(seq) FROM chat_messages
                             WHERE user_id = %s AND session_id = %s), 0)
                       + row_number() OVER (),
                   t.role, t.content
            FROM unnest(%s::text[], %s::text[]) AS t(role, content)
            """,
            (user_id, session_id, user_id, session_id,
             [r[0] for r in rows], [r[1] for r in rows]),
        )


async def getRecentMessages(user_id: str, session_id: str, limit: int) -> list[dict]:
    """Last `limit` turns, oldest first. Used to warm the cache on a miss."""
    async with get_cursor() as cur:
        await cur.execute(
            """
            SELECT role, content FROM chat_messages
            WHERE user_id = %s AND session_id = %s
            ORDER BY seq DESC LIMIT %s
            """,
            (user_id, session_id, limit),
        )
        return list(reversed(await cur.fetchall()))


async def getMessagesBefore(user_id: str, session_id: str, before_seq: int | None,
                      limit: int = 50) -> list[dict]:
    """One page of older turns, newest first. Cursor-paginated on seq, never OFFSET."""
    async with get_cursor() as cur:
        if before_seq is None:
            await cur.execute(
                """
                SELECT seq, role, content FROM chat_messages
                WHERE user_id = %s AND session_id = %s
                ORDER BY seq DESC LIMIT %s
                """,
                (user_id, session_id, limit),
            )
        else:
            await cur.execute(
                """
                SELECT seq, role, content FROM chat_messages
                WHERE user_id = %s AND session_id = %s AND seq < %s
                ORDER BY seq DESC LIMIT %s
                """,
                (user_id, session_id, before_seq, limit),
            )
        return await cur.fetchall()


# --- partition maintenance ------------------------------------------------

async def listMonthPartitions() -> list[dict]:
    """Every chat_messages month partition, oldest first."""
    async with get_cursor() as cur:
        await cur.execute(
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
        return await cur.fetchall()


async def detachMonthPartitions(before: datetime) -> list[str]:
    """Detach month partitions older than `before`. They survive as standalone tables:
    export, then drop."""
    cutoff = before.strftime("%Y_%m")
    # Listed before opening the write cursor: nesting would take a second connection.
    stale = [r["name"] for r in await listMonthPartitions() if r["name"][-7:] < cutoff]

    async with get_cursor(commit=True) as cur:
        for name in stale:
            parent = name[:name.rindex("_", 0, name.rindex("_"))]
            await cur.execute(f"ALTER TABLE {parent} DETACH PARTITION {name}")
    return stale
