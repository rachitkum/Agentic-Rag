"""
Weaviate vector store for the RAG agent.

We supply our own vectors (local MiniLM embeddings), so the collection is created with
vectorizer = none. Every RAPTOR node (leaf chunk or cluster summary) is one object.

The collection is multi-tenant: a tenant is a bucket of users (src/db/tenancy.py), so a
search enters only that bucket's index. Within it, `user_id` scopes to one user.

Run Weaviate on Docker; connect via env:
    WEAVIATE_URL       (e.g. http://localhost:8080)
    WEAVIATE_API_KEY   (optional; only if auth is enabled)
"""

import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

COLLECTION_NAME = "RagNode"

_CLIENT = None


def _connect():
    """Open a Weaviate client against the Docker instance."""
    # weaviate-client v4 wants host/port split out
    from urllib.parse import urlparse
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"

    if WEAVIATE_API_KEY:
        return weaviate.connect_to_custom(
            http_host=host, http_port=port, http_secure=secure,
            grpc_host=host, grpc_port=50051, grpc_secure=secure,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        )
    return weaviate.connect_to_custom(
        http_host=host, http_port=port, http_secure=secure,
        grpc_host=host, grpc_port=50051, grpc_secure=secure,
    )


def ensure_schema(client) -> None:
    """Create the multi-tenant RagNode collection if it doesn't exist."""
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        # auto_tenant_creation lets a bucket's first upload create it inline.
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=True,
            auto_tenant_activation=True,
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="node_type", data_type=DataType.TEXT),   # "leaf" | "summary"
            Property(name="level", data_type=DataType.INT),
            Property(name="user_id", data_type=DataType.TEXT),     # per-user isolation
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="session_id", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="file_name", data_type=DataType.TEXT),
        ],
    )


def get_client():
    """Return the shared client, connecting on first use. Do not close it."""
    global _CLIENT
    if _CLIENT is None or not _CLIENT.is_connected():
        _CLIENT = _connect()
        ensure_schema(_CLIENT)
    return _CLIENT


def close_client() -> None:
    """Close the shared client (called on app shutdown)."""
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def _tenant(client, tenant_id: str):
    return client.collections.get(COLLECTION_NAME).with_tenant(tenant_id)


def insert_nodes(client, nodes: list[dict], tenant_id: str, user_id: str,
                 doc_id: str, session_id: str = "") -> int:
    """Batch-insert one document's RAPTOR nodes into its owner's tenant."""
    collection = _tenant(client, tenant_id)
    with collection.batch.dynamic() as batch:
        for node in nodes:
            vector = node["embedding"]
            batch.add_object(
                properties={
                    "text": node["text"],
                    "node_type": node["node_type"],
                    "level": node["level"],
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "source": node.get("source", ""),
                    "file_name": node.get("file_name", ""),
                },
                vector=vector,
            )
    return len(nodes)


def search(client, query_vector, tenant_id: str, user_id: str, limit: int = 50,
           node_type: str = None) -> list[dict]:
    """Vector search in one tenant, scoped to one user, optionally by node type."""
    collection = _tenant(client, tenant_id)

    filters = Filter.by_property("user_id").equal(user_id)
    if node_type:
        filters = filters & Filter.by_property("node_type").equal(node_type)

    res = collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        filters=filters,
        return_properties=["text", "node_type", "level", "source", "file_name"],
    )
    out = []
    for obj in res.objects:
        p = obj.properties
        out.append({
            "text": p.get("text"),
            "node_type": p.get("node_type", "leaf"),
            "level": p.get("level"),
            "source": p.get("source"),
            "file_name": p.get("file_name"),
        })
    return out
