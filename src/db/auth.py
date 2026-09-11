"""
User identity and tenant resolution.

No password: POST /user is both signup and login. resolveTenant() sits on every read
and write path, so it is cached in-process.
"""

from src.db import postgres, tenancy

_TENANT_CACHE: dict[str, str] = {}
_CACHE_MAX = 100_000


async def createOrLoginUser(user_id: str) -> dict:
    """Register a user, or return them if they already exist."""
    existing = await postgres.getUserRow(user_id)
    if existing is not None:
        _TENANT_CACHE[user_id] = existing["tenant_id"]
        return {**existing, "created": False}

    tenant_id = await tenancy.computeBucket(user_id)
    row = await postgres.insertUser(user_id, tenant_id)
    _TENANT_CACHE[user_id] = row["tenant_id"]
    return {**row, "created": True}


async def resolveTenant(user_id: str) -> str | None:
    """Tenant for a user, or None if unknown. Called on every upload and chat turn."""
    cached = _TENANT_CACHE.get(user_id)
    if cached is not None:
        return cached

    row = await postgres.getUserRow(user_id)
    if row is None:
        return None

    if len(_TENANT_CACHE) >= _CACHE_MAX:
        _TENANT_CACHE.clear()
    _TENANT_CACHE[user_id] = row["tenant_id"]
    return row["tenant_id"]
