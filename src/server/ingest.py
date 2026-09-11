"""
Document ingestion.

Chunk a PDF, build a RAPTOR tree (cluster similar chunks -> LLM-summarize each
cluster -> repeat), and store every node (leaves + summaries) in the user's Weaviate
tenant. Retrieval (KB.py) searches leaves and summaries together, so broad questions
land on summary nodes and specific ones on leaf chunks.

session_id is recorded for provenance only; it does not scope retrieval.
"""

import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.server.utils import getEmbeddings
from src.server.LLM import ChatOpenAI
from src.server.raptor import build_raptor_tree
from src.db import vectorstore


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


def ingest_pdf(pdf_bytes: bytes, user_id: str, tenant_id: str, doc_id: str,
               session_id: str = "", file_name: str = "") -> dict:
    """Chunk -> build RAPTOR tree -> store all nodes in the owner's tenant."""
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
    vectorstore.insert_nodes(client, nodes, tenant_id, user_id, doc_id, session_id)

    leaves = sum(1 for n in nodes if n["node_type"] == "leaf")
    summaries = len(nodes) - leaves
    print(f"Ingested '{file_name}' for {user_id} in {tenant_id}: {len(nodes)} nodes ({leaves} leaves, {summaries} summaries)")
    return {"chunks": len(chunks), "nodes": len(nodes), "leaves": leaves, "summaries": summaries}
