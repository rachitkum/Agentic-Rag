# short- term memory last 6 convos for llm context of chat
import json
import os
import threading

from dotenv import load_dotenv
import valkey

from src.db import postgres

load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")

# Turns kept in the prompt; the list is trimmed to this on append.
TURN_WINDOW = 6
TTL_SECONDS = 60 * 60 * 24 * 7

_CLIENT = None

# Serialises first connect: these calls reach here from threadpool workers, so an
# unlocked check-then-assign lets concurrent callers each build a client and orphan
# all but the last.
_CLIENT_LOCK = threading.Lock()


def get_client():
    """Return the shared Valkey client, connecting on first use.

    Double-checked: the common path takes no lock; the re-check inside means callers
    that queued behind the first connect reuse the client it built.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = valkey.Valkey.from_url(VALKEY_URL, decode_responses=True)
        return _CLIENT


def close_client() -> None:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None


def _key(user_id: str, session_id: str) -> str:
    return f"chat:{user_id}:{session_id}"


async def getHistory(user_id: str, session_id: str, limit: int = TURN_WINDOW) -> list[dict]:
    """Last `limit` turns, oldest first. Falls back to Postgres and warms the cache."""
    if not user_id or not session_id:
        return []

    try:
        raw = get_client().lrange(_key(user_id, session_id), -limit, -1)
        if raw:
            return [json.loads(t) for t in raw]
    except Exception as e:
        print("ERROR reading chat history from cache:", e)

    # Miss: new session, evicted, expired, or restarted.
    try:
        turns = await postgres.getRecentMessages(user_id, session_id, limit)
    except Exception as e:
        print("ERROR reading chat history from postgres:", e)
        return []

    if turns:
        _warm(user_id, session_id, turns)
    return turns


def _warm(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Repopulate the cache from the durable log."""
    try:
        key = _key(user_id, session_id)
        pipe = get_client().pipeline()
        pipe.delete(key)
        pipe.rpush(key, *[json.dumps(t) for t in turns])
        pipe.expire(key, TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        print("ERROR warming chat history:", e)


async def appendTurns(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Postgres first, then cache: a lost cache entry is recoverable, a lost write isn't."""
    if not user_id or not session_id or not turns:
        return

    try:
        await postgres.appendMessages(user_id, session_id, turns)
    except Exception as e:
        print("ERROR writing chat history to postgres:", e)

    try:
        key = _key(user_id, session_id)
        pipe = get_client().pipeline()
        pipe.rpush(key, *[json.dumps(t) for t in turns])
        pipe.ltrim(key, -TURN_WINDOW, -1)
        pipe.expire(key, TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        print("ERROR writing chat history to cache:", e)


# --- ingestion jobs -------------------------------------------------------
# An upload returns a job_id immediately and processes the PDF in the background, so
# /upload/status is the only way the client learns whether it finished or failed.
# Held in a process dict this caps the app at one replica: a poll load-balanced to
# another replica finds nothing and 404s while the job runs fine elsewhere.
#
# Unlike the chat cache above there is nothing durable underneath, so these writes
# RAISE rather than swallow -- a lost job record leaves the user with no way to see
# the outcome, including a failure.

# Long enough for a client to poll a slow ingest through to completion.
JOB_TTL_SECONDS = 60 * 60 * 24


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _writeJob(job_id: str, payload: dict) -> None:
    get_client().set(_job_key(job_id), json.dumps(payload), ex=JOB_TTL_SECONDS)


def setJobProcessing(job_id: str) -> None:
    """Record a job as started. Raises if the write fails."""
    _writeJob(job_id, {"status": "processing"})


def setJobDone(job_id: str, result: dict) -> None:
    _writeJob(job_id, {"status": "done", "result": result})


def setJobError(job_id: str, error: str) -> None:
    _writeJob(job_id, {"status": "error", "error": error})


def getJob(job_id: str) -> dict | None:
    """Job record, or None if unknown or expired."""
    raw = get_client().get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)
