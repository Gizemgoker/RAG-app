"""Retrieve relevant document chunks from the local ChromaDB collection."""

from typing import Any

from src.embeddings import get_embeddings
from src.vector_store import TXT_PAGE_SENTINEL, initialize_collection


def retrieve_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return up to ``top_k`` chunks most similar to a non-empty query."""
    if not query.strip() or top_k <= 0:
        return []

    collection = initialize_collection()
    collection_size = collection.count()
    if not collection_size:
        return []

    query_embeddings = get_embeddings([query])
    if not query_embeddings:
        return []

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=min(top_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": document,
            "source": metadata["source"],
            "page": None if metadata["page"] == TXT_PAGE_SENTINEL else metadata["page"],
            "chunk_id": metadata["chunk_id"],
            "distance": distance,
        }
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
