import pytest

from src.vector_store import TXT_PAGE_SENTINEL, initialize_collection, store_chunks


def test_stores_chunks_and_metadata_in_persistent_collection(tmp_path):
    collection = initialize_collection(tmp_path / "chroma_db")
    chunks = [
        {"text": "text file content", "source": "notes.txt", "page": None, "chunk_id": 0},
        {"text": "pdf page content", "source": "guide.pdf", "page": 1, "chunk_id": 0},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    store_chunks(collection, chunks, embeddings)

    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    records = {
        metadata["source"]: (document, metadata, embedding)
        for document, metadata, embedding in zip(
            stored["documents"], stored["metadatas"], stored["embeddings"]
        )
    }
    notes_document, notes_metadata, notes_embedding = records["notes.txt"]
    assert notes_document == "text file content"
    assert notes_metadata == {"source": "notes.txt", "page": TXT_PAGE_SENTINEL, "chunk_id": 0}
    assert notes_embedding.tolist() == pytest.approx([0.1, 0.2])

    pdf_document, pdf_metadata, pdf_embedding = records["guide.pdf"]
    assert pdf_document == "pdf page content"
    assert pdf_metadata == {"source": "guide.pdf", "page": 1, "chunk_id": 0}
    assert pdf_embedding.tolist() == pytest.approx([0.3, 0.4])


def test_reindexing_same_chunks_does_not_create_duplicates(tmp_path):
    collection = initialize_collection(tmp_path / "chroma_db")
    chunks = [{"text": "content", "source": "notes.txt", "page": None, "chunk_id": 0}]
    embeddings = [[0.1, 0.2]]

    store_chunks(collection, chunks, embeddings)
    first_ids = collection.get()["ids"]
    store_chunks(collection, chunks, embeddings)

    assert collection.count() == 1
    assert collection.get()["ids"] == first_ids


def test_store_chunks_validates_parallel_input_lengths(tmp_path):
    collection = initialize_collection(tmp_path / "chroma_db")

    try:
        store_chunks(
            collection,
            [{"text": "content", "source": "notes.txt", "page": None, "chunk_id": 0}],
            [],
        )
    except ValueError as error:
        assert str(error) == "chunks and embeddings must have the same length"
    else:
        raise AssertionError("Expected mismatched chunks and embeddings to raise ValueError")
