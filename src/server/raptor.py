"""
RAPTOR-style hierarchical clustering for advanced RAG.

Problem it solves: plain top-k retrieval breaks on big documents. A "summarize the
whole thing" query is not similar to any single chunk, so top-k over-samples one
theme and silently drops the rest.

Fix: recursively cluster chunks by embedding similarity, LLM-summarize each cluster
into a higher-level node, cluster those, and repeat. This builds a tree:

        [root summary]            <- broadest
        /     |      \\
    [sum]   [sum]   [sum]         <- mid-level themes
    / \\     / \\     / \\
   c   c   c   c   c   c          <- raw leaf chunks (detail)

We store EVERY node (leaves + all summaries) in the same Weaviate collection with a
`node_type` ("leaf"/"summary") and `level`. Retrieval then does a single vector
search over the flattened tree ("collapsed tree" retrieval):

  - specific query  -> naturally matches leaf nodes (detail)
  - broad/summary Q -> naturally matches summary nodes (coverage), because a
                       summary node's embedding already represents a whole theme

So one index, one query path handles both small and large docs, and both broad and
specific questions, without missing data.

This module only BUILDS the tree from a list of chunks. Insertion into Weaviate
lives in ingest.py; runtime retrieval lives in KB.py.
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
    """
    Build the hierarchical node list from raw leaf chunks.

    Args:
        chunks:       raw text chunks (the leaves).
        embed_fn:     text -> np.ndarray embedding.
        summarize_fn: list[str] -> str, LLM summary of a cluster of texts.

    Returns:
        list of node dicts, each:
          { "text": str, "embedding": list[float], "node_type": "leaf"|"summary",
            "level": int }
        Level 0 = leaves; higher levels = summaries. Ready to store in Weaviate.
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
