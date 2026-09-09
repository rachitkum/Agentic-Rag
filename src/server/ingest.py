"""
In-session document ingestion.

A user uploads a PDF inside their chat session. We chunk it, build a RAPTOR tree
(cluster similar chunks -> LLM-summarize each cluster -> repeat), and store every
node (leaves + summaries) in Weaviate scoped to that session_id. Retrieval (KB.py)
then searches leaves and summaries together, so "summarize the whole document"
lands on high-level summary nodes and a specific question lands on leaf chunks —
one collection, no missed data — and only within the user's own session.
"""

import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.server.utils import getEmbeddings
from src.server.LLM import ChatOpenAI
from src.server.raptor import build_raptor_tree
from src.server import vectorstore


def chunk_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """Split raw PDF bytes into overlapping text chunks (leaf nodes)."""
    tmp = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.close()
        loader = PyMuPDFLoader(tmp.name)
        docs = loader.load()
    finally:
        os.unlink(tmp.name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return [d.page_content for d in splitter.split_documents(docs)]


def ingest_pdf(pdf_bytes: bytes, session_id: str, file_name: str = "") -> dict:
    """Chunk -> build RAPTOR tree -> store all nodes in Weaviate for this session."""
    chunks = chunk_pdf_bytes(pdf_bytes)
    if not chunks:
        return {"chunks": 0, "nodes": 0, "leaves": 0, "summaries": 0}

    llm = ChatOpenAI()
    nodes = build_raptor_tree(
        chunks=chunks,
        embed_fn=getEmbeddings,
        summarize_fn=llm.summarizeCluster,
    )

    for node in nodes:
        node["source"] = file_name
        node["file_name"] = file_name

    client = vectorstore.get_client()
    try:
        vectorstore.insert_nodes(client, nodes, session_id)
    finally:
        client.close()

    leaves = sum(1 for n in nodes if n["node_type"] == "leaf")
    summaries = len(nodes) - leaves
    print(f"Ingested '{file_name}' into session {session_id}: {len(nodes)} nodes ({leaves} leaves, {summaries} summaries)")
    return {"chunks": len(chunks), "nodes": len(nodes), "leaves": leaves, "summaries": summaries}
