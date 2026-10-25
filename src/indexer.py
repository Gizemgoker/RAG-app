"""Coordinate document loading, embedding generation, and vector storage."""

from pathlib import Path

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.vector_store import initialize_collection, store_chunks


def index_documents(
    documents_dir: str | Path = "documents",
    persist_directory: str | Path = "chroma_db",
) -> int:
    """Index local documents and return the number of stored chunks."""
    chunks = load_documents(documents_dir)
    if not chunks:
        return 0

    embeddings = get_embeddings([chunk["text"] for chunk in chunks])
    collection = initialize_collection(persist_directory)
    store_chunks(collection, chunks, embeddings)
    return len(chunks)
