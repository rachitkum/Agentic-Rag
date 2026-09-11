"""
Tenant assignment.

A tenant is a bucket of users. At signup a user is hashed into a bucket and that value
is stored in Postgres; reads always use the stored value, so growing the bucket count
never remaps existing users.
"""

import hashlib

# ~250 users x ~10 docs x ~65 nodes = ~160k vectors per shard.
USERS_PER_BUCKET = 250
MIN_BUCKETS = 16


def _next_power_of_two(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def bucketCountFor(user_count: int) -> int:
    """How many buckets a user base of this size should spread across."""
    target = -(-user_count // USERS_PER_BUCKET)
    return max(MIN_BUCKETS, _next_power_of_two(target))


async def currentBucketCount() -> int:
    """Bucket count for assigning a new user, from the live user count."""
    from src.db import postgres
    return bucketCountFor(await postgres.getUserCount())


def _hash(user_id: str) -> int:
    """Stable 64-bit hash (builtin hash() is salted per process)."""
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


async def computeBucket(user_id: str, bucket_count: int | None = None) -> str:
    """Bucket for a user at signup. Not a lookup -- use auth.resolveTenant() for that."""
    if bucket_count is None:
        bucket_count = await currentBucketCount()
    return f"bucket_{_hash(user_id) % bucket_count:05d}"


def dedicatedTenant(user_id: str) -> str:
    """Tenant name for a user promoted out of a shared bucket."""
    return f"user_{user_id}"


def isDedicated(tenant_id: str) -> bool:
    """True if this tenant holds exactly one user."""
    return tenant_id.startswith("user_")
