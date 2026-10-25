from unittest.mock import Mock, patch

from src.retriever import retrieve_chunks


def test_retrieve_chunks_embeds_query_and_returns_results():
    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["text chunk", "pdf chunk"]],
        "metadatas": [[
            {"source": "notes.txt", "page": -1, "chunk_id": 0},
            {"source": "guide.pdf", "page": 2, "chunk_id": 1},
        ]],
        "distances": [[0.1, 0.3]],
    }

    with (
        patch("src.retriever.initialize_collection", return_value=collection) as initialize_collection,
        patch("src.retriever.get_embeddings", return_value=[[0.1, 0.2]]) as get_embeddings,
    ):
        chunks = retrieve_chunks("What is in the guide?", top_k=3)

    assert chunks == [
        {
            "text": "text chunk",
            "source": "notes.txt",
            "page": None,
            "chunk_id": 0,
            "distance": 0.1,
        },
        {
            "text": "pdf chunk",
            "source": "guide.pdf",
            "page": 2,
            "chunk_id": 1,
            "distance": 0.3,
        },
    ]
    initialize_collection.assert_called_once_with()
    get_embeddings.assert_called_once_with(["What is in the guide?"])
    collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )


def test_retrieve_chunks_skips_empty_query_without_openai_call():
    with (
        patch("src.retriever.initialize_collection") as initialize_collection,
        patch("src.retriever.get_embeddings") as get_embeddings,
    ):
        chunks = retrieve_chunks("   ")

    assert chunks == []
    initialize_collection.assert_not_called()
    get_embeddings.assert_not_called()


def test_retrieve_chunks_skips_embedding_for_empty_collection():
    collection = Mock()
    collection.count.return_value = 0

    with (
        patch("src.retriever.initialize_collection", return_value=collection),
        patch("src.retriever.get_embeddings") as get_embeddings,
    ):
        chunks = retrieve_chunks("question")

    assert chunks == []
    get_embeddings.assert_not_called()
    collection.query.assert_not_called()
