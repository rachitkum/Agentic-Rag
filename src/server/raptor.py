"""
RAPTOR-style hierarchical clustering for advanced RAG.

Plain top-k breaks on big documents: a "summarize the whole thing" query matches no
single chunk, so it over-samples one theme and drops the rest. Fix: recursively
cluster chunks by embedding similarity, LLM-summarize each cluster into a higher-level
node, and repeat.

        [root summary]            <- broadest
        /     |      \\
    [sum]   [sum]   [sum]         <- mid-level themes
    / \\     / \\     / \\
   c   c   c   c   c   c          <- raw leaf chunks (detail)

Every node is stored in one Weaviate collection with node_type and level, so a single
search over the flattened tree serves both question types: specific queries match
leaves, broad ones match summaries.

This module only builds the tree. Insertion lives in ingest.py, retrieval in KB.py.
"""

import numpy as np
from sklearn.mixture import GaussianMixture


# ---- tunables -------------------------------------------------------------
MAX_LEVELS = 3            # how deep the tree can grow
MIN_CLUSTER_SIZE = 3      # stop recursing a branch smaller than this
MAX_CLUSTERS = 12         # cap components tried when picking cluster count
RANDOM_STATE = 42         # deterministic clustering


def _best_cluster_count(embeddings: np.ndarray, max_clusters: int) -> int:
    """Pick the number of clusters via BIC (favours the simplest good fit)."""
    n = len(embeddings)
    if n <= MIN_CLUSTER_SIZE:
        return 1
    max_k = min(max_clusters, n)
    best_k, best_bic = 1, float("inf")
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
        gm.fit(embeddings)
        bic = gm.bic(embeddings)
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k


def _cluster(embeddings: np.ndarray) -> list[list[int]]:
    """Return groups of indices, one list per cluster."""
    k = _best_cluster_count(embeddings, MAX_CLUSTERS)
    if k <= 1:
        return [list(range(len(embeddings)))]
    gm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    labels = gm.fit_predict(embeddings)
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)
    return list(groups.values())


def build_raptor_tree(chunks: list[str], embed_fn, summarize_fn) -> list[dict]:
    """Build the node list from raw leaf chunks.

    Returns {"text", "embedding", "node_type", "level"} dicts. Level 0 = leaves,
    higher levels = summaries.
    """
    nodes: list[dict] = []

    # Level 0 — the leaves themselves.
    current_texts = list(chunks)
    current_embeddings = np.array([embed_fn(t) for t in current_texts])
    for text, emb in zip(current_texts, current_embeddings):
        nodes.append({
            "text": text,
            "embedding": emb.tolist(),
            "node_type": "leaf",
            "level": 0,
        })

    # Higher levels — cluster, summarize, repeat.
    level = 1
    while level <= MAX_LEVELS and len(current_texts) > MIN_CLUSTER_SIZE:
        groups = _cluster(current_embeddings)

        # A single cluster covering everything means the tree has converged.
        if len(groups) <= 1:
            summary = summarize_fn(current_texts)
            emb = embed_fn(summary)
            nodes.append({
                "text": summary,
                "embedding": emb.tolist(),
                "node_type": "summary",
                "level": level,
            })
            break

        next_texts, next_embeddings = [], []
        for group in groups:
            cluster_texts = [current_texts[i] for i in group]
            summary = summarize_fn(cluster_texts)
            emb = embed_fn(summary)
            nodes.append({
                "text": summary,
                "embedding": emb.tolist(),
                "node_type": "summary",
                "level": level,
            })
            next_texts.append(summary)
            next_embeddings.append(emb)

        current_texts = next_texts
        current_embeddings = np.array(next_embeddings)
        level += 1

    print(f"RAPTOR tree built: {len(nodes)} nodes across {level} level(s)")
    return nodes
