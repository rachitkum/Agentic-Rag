"""
Weaviate vector store for the RAG agent.

We supply our own vectors (local MiniLM embeddings), so the Weaviate collection is
created with vectorizer = none. Every RAPTOR node (leaf chunk or cluster summary)
is one object, scoped to a chat session via `session_id` so a user only ever
retrieves from the documents they uploaded in that session.

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
    """Create the RagNode collection if it doesn't exist (vectorizer = none)."""
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="node_type", data_type=DataType.TEXT),   # "leaf" | "summary"
            Property(name="level", data_type=DataType.INT),
            Property(name="session_id", data_type=DataType.TEXT),  # per-chat isolation
            Property(name="source", data_type=DataType.TEXT),
            Property(name="file_name", data_type=DataType.TEXT),
        ],
    )


def get_client():
    """Return a connected client with the schema ensured. Caller must close it."""
    client = _connect()
    ensure_schema(client)
    return client


def insert_nodes(client, nodes: list[dict], session_id: str) -> int:
    """Batch-insert RAPTOR nodes for a session. Each node dict has text, embedding,
    node_type, level, source, file_name (embedding is popped out as the vector)."""
    collection = client.collections.get(COLLECTION_NAME)
    with collection.batch.dynamic() as batch:
        for node in nodes:
            vector = node["embedding"]
            batch.add_object(
                properties={
                    "text": node["text"],
                    "node_type": node["node_type"],
                    "level": node["level"],
                    "session_id": session_id,
                    "source": node.get("source", ""),
                    "file_name": node.get("file_name", ""),
                },
                vector=vector,
            )
    return len(nodes)


def search(client, query_vector, session_id: str, limit: int = 50, node_type: str = None) -> list[dict]:
    """Vector search scoped to a session, optionally restricted to leaf/summary nodes."""
    collection = client.collections.get(COLLECTION_NAME)

    filters = Filter.by_property("session_id").equal(session_id)
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


def delete_session(client, session_id: str) -> None:
    """Remove every node belonging to a session (e.g. when the chat ends)."""
    collection = client.collections.get(COLLECTION_NAME)
    collection.data.delete_many(
        where=Filter.by_property("session_id").equal(session_id)
    )
