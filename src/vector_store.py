"""Persistent local ChromaDB storage for document chunks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb


DEFAULT_COLLECTION_NAME = "documents"
TXT_PAGE_SENTINEL = -1


def initialize_collection(
    persist_directory: str | Path = "chroma_db",
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Return a persistent ChromaDB collection, creating it when needed."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(directory))
    return client.get_or_create_collection(name=collection_name)


def store_chunks(collection, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
    """Upsert chunks and their precomputed embeddings into a collection."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        return

    collection.upsert(
        ids=[_chunk_record_id(chunk) for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[_chunk_metadata(chunk) for chunk in chunks],
    )


def _chunk_record_id(chunk: dict[str, Any]) -> str:
    """Create a stable ID from metadata that is unique within the documents set."""
    page = chunk["page"] if chunk["page"] is not None else TXT_PAGE_SENTINEL
    identity = f"{chunk['source']}\0{page}\0{chunk['chunk_id']}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert chunk metadata to ChromaDB-compatible scalar values."""
    return {
        "source": chunk["source"],
        "page": chunk["page"] if chunk["page"] is not None else TXT_PAGE_SENTINEL,
        "chunk_id": chunk["chunk_id"],
    }
