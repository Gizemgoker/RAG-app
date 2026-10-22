from unittest.mock import patch

import pytest

from src.document_loader import chunk_text, load_documents


def test_loads_utf8_text_file_with_metadata(tmp_path):
    (tmp_path / "notes.txt").write_text("caffè", encoding="utf-8")

    chunks = load_documents(tmp_path)

    assert chunks == [
        {"text": "caffè", "source": "notes.txt", "page": None, "chunk_id": 0}
    ]


def test_pdf_chunks_include_one_based_page_metadata(tmp_path):
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.write_bytes(b"placeholder")

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        pages = [Page("first page"), Page("second page")]

    with patch("src.document_loader.PdfReader", return_value=Reader()):
        chunks = load_documents(tmp_path)

    assert [(chunk["text"], chunk["page"], chunk["chunk_id"]) for chunk in chunks] == [
        ("first page", 1, 0),
        ("second page", 2, 1),
    ]


def test_chunks_respect_size_and_overlap():
    chunks = chunk_text("abcdefghij", "sample.txt", None, chunk_size=4, chunk_overlap=1)

    assert [chunk["text"] for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert all(len(chunk["text"]) <= 4 for chunk in chunks)


def test_stops_after_a_chunk_reaches_the_end_of_text():
    chunks = chunk_text("abcdef", "sample.txt", None, chunk_size=4, chunk_overlap=2)

    assert [chunk["text"] for chunk in chunks] == ["abcd", "cdef"]


def test_empty_input_produces_no_chunks(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    assert chunk_text("", "empty.txt", None) == []
    assert load_documents(tmp_path) == []


@pytest.mark.parametrize("chunk_size, chunk_overlap", [(0, 0), (-1, 0), (10, -1), (10, 10)])
def test_invalid_chunk_parameters_raise_value_error(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("text", "sample.txt", None, chunk_size, chunk_overlap)


def test_chunking_terminates_when_overlap_is_one_less_than_size():
    chunks = chunk_text("abcdef", "sample.txt", None, chunk_size=2, chunk_overlap=1)

    assert [chunk["text"] for chunk in chunks] == ["ab", "bc", "cd", "de", "ef"]
