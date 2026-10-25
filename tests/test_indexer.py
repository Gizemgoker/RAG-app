from unittest.mock import Mock, patch

from src.indexer import index_documents


def test_index_documents_runs_each_pipeline_step(tmp_path):
    chunks = [
        {"text": "first", "source": "notes.txt", "page": None, "chunk_id": 0},
        {"text": "second", "source": "notes.txt", "page": None, "chunk_id": 1},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    collection = Mock()

    with (
        patch("src.indexer.load_documents", return_value=chunks) as load_documents,
        patch("src.indexer.get_embeddings", return_value=embeddings) as get_embeddings,
        patch("src.indexer.initialize_collection", return_value=collection) as initialize_collection,
        patch("src.indexer.store_chunks") as store_chunks,
    ):
        indexed_count = index_documents(tmp_path / "documents", tmp_path / "chroma_db")

    assert indexed_count == 2
    load_documents.assert_called_once_with(tmp_path / "documents")
    get_embeddings.assert_called_once_with(["first", "second"])
    initialize_collection.assert_called_once_with(tmp_path / "chroma_db")
    store_chunks.assert_called_once_with(collection, chunks, embeddings)


def test_index_documents_skips_embeddings_for_no_chunks(tmp_path):
    with (
        patch("src.indexer.load_documents", return_value=[]) as load_documents,
        patch("src.indexer.get_embeddings") as get_embeddings,
        patch("src.indexer.initialize_collection") as initialize_collection,
        patch("src.indexer.store_chunks") as store_chunks,
    ):
        indexed_count = index_documents(tmp_path / "documents", tmp_path / "chroma_db")

    assert indexed_count == 0
    load_documents.assert_called_once_with(tmp_path / "documents")
    get_embeddings.assert_not_called()
    initialize_collection.assert_not_called()
    store_chunks.assert_not_called()
