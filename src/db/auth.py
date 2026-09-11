"""
User identity and tenant resolution.

No password: POST /user is both signup and login. A known user_id returns the existing
row and its existing tenant; an unknown one is created and assigned a bucket.

resolveTenant() sits on every read and write path, so it is cached in-process. A user's
tenant only changes during a promotion, which is rare and offline.
"""

from src.db import postgres, tenancy

_TENANT_CACHE: dict[str, str] = {}
_CACHE_MAX = 100_000


def createOrLoginUser(user_id: str) -> dict:
    """Register a user, or return them if they already exist."""
    existing = postgres.getUserRow(user_id)
    if existing is not None:
        _TENANT_CACHE[user_id] = existing["tenant_id"]
        return {**existing, "created": False}

    row = postgres.insertUser(user_id, tenancy.computeBucket(user_id))
    _TENANT_CACHE[user_id] = row["tenant_id"]
    return {**row, "created": True}


def getUser(user_id: str) -> dict | None:
    return postgres.getUserRow(user_id)


def resolveTenant(user_id: str) -> str | None:
    """Tenant for a user, or None if unknown. Called on every upload and chat turn."""
    cached = _TENANT_CACHE.get(user_id)
    if cached is not None:
        return cached

    row = postgres.getUserRow(user_id)
    if row is None:
        return None

    if len(_TENANT_CACHE) >= _CACHE_MAX:
        _TENANT_CACHE.clear()
    _TENANT_CACHE[user_id] = row["tenant_id"]
    return row["tenant_id"]


def invalidateTenant(user_id: str) -> None:
    """Drop a cached mapping. Called after a promotion moves a user."""
    _TENANT_CACHE.pop(user_id, None)
