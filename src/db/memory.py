# short- term memory last 6 convos for llm context of chat
import json
import os

from dotenv import load_dotenv
import valkey

from src.db import postgres

load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")

# Turns kept in the prompt; the list is trimmed to this on append.
TURN_WINDOW = 6
TTL_SECONDS = 60 * 60 * 24 * 7

_CLIENT = None


def get_client():
    """Return the shared Valkey client, connecting on first use."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = valkey.Valkey.from_url(VALKEY_URL, decode_responses=True)
    return _CLIENT


def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def _key(user_id: str, session_id: str) -> str:
    return f"chat:{user_id}:{session_id}"


def getHistory(user_id: str, session_id: str, limit: int = TURN_WINDOW) -> list[dict]:
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
        turns = postgres.getRecentMessages(user_id, session_id, limit)
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


def appendTurns(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Postgres first, then cache: a lost cache entry is recoverable, a lost write isn't."""
    if not user_id or not session_id or not turns:
        return

    try:
        postgres.appendMessages(user_id, session_id, turns)
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
