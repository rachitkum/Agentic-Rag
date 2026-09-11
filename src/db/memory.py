"""
Short-term conversation memory in Valkey.

One list per (user_id, session_id) holding the recent turns of that chat. The client
no longer sends history: it sends user_id + session_id, and the server reads the last
TURN_WINDOW turns, then appends the new user message and the assistant reply.

Keys expire after TTL_SECONDS so abandoned chats clean themselves up.

Env:
    VALKEY_URL   (e.g. redis://localhost:6379/0)
"""

import json
import os

from dotenv import load_dotenv
import valkey

load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")

# Turns kept in the prompt. The list is trimmed to this on every append.
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
    """Last `limit` turns, oldest first. Empty list for a new or expired session."""
    if not user_id or not session_id:
        return []
    try:
        raw = get_client().lrange(_key(user_id, session_id), -limit, -1)
        return [json.loads(t) for t in raw]
    except Exception as e:
        print("ERROR reading chat history:", e)
        return []


def appendTurns(user_id: str, session_id: str, turns: list[dict]) -> None:
    """Append turns, trim to the window, and refresh the TTL."""
    if not user_id or not session_id or not turns:
        return
    try:
        key = _key(user_id, session_id)
        pipe = get_client().pipeline()
        pipe.rpush(key, *[json.dumps(t) for t in turns])
        pipe.ltrim(key, -TURN_WINDOW, -1)
        pipe.expire(key, TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        print("ERROR writing chat history:", e)
