import io
from unittest.mock import patch

from app import app


def test_index_page_renders_question_form():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b'name="question"' in response.data
    assert b"Ask" in response.data
    assert b"Finding an answer" in response.data


def test_question_submission_displays_answer():
    client = app.test_client()

    with patch("app.answer_question", return_value="The answer is in the document.") as answer_question:
        response = client.post("/", data={"question": "What is the answer?"})

    assert response.status_code == 200
    assert b"The answer is in the document." in response.data
    answer_question.assert_called_once_with("What is the answer?")


def test_ask_form_targets_question_endpoint_and_never_upload_endpoint():
    client = app.test_client()

    with (
        patch("app.answer_question", return_value="Answer") as answer_question,
        patch("app.upload_document") as upload_document,
    ):
        response = client.post("/", data={"question": "Question"})

    assert response.status_code == 200
    assert b'<form id="question-form" method="post" action="/">' in response.data
    assert b'<form id="upload-form" method="post" action="/upload"' in response.data
    answer_question.assert_called_once_with("Question")
    upload_document.assert_not_called()


def test_question_submission_handles_errors_without_exposing_details():
    client = app.test_client()

    with patch("app.answer_question", side_effect=RuntimeError("sensitive technical detail")):
        response = client.post("/", data={"question": "What is the answer?"})

    assert response.status_code == 200
    assert b"Unable to answer your question right now. Please try again." in response.data
    assert b"sensitive technical detail" not in response.data


def test_blank_question_does_not_call_answer_generation():
    client = app.test_client()

    with patch("app.answer_question") as answer_question:
        response = client.post("/", data={"question": "   "})

    assert b"Please enter a question." in response.data
    answer_question.assert_not_called()


def test_upload_indexes_one_valid_document(tmp_path, monkeypatch):
    monkeypatch.setitem(app.config, "DOCUMENTS_DIRECTORY", tmp_path)
    client = app.test_client()

    with patch("app.index_file", return_value=2) as index_file:
        response = client.post(
            "/upload",
            data={"document": (io.BytesIO(b"upload content"), "report.txt")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"Uploaded and indexed 2 chunks." in response.data
    index_file.assert_called_once_with(tmp_path / "report.txt")
    assert (tmp_path / "report.txt").read_bytes() == b"upload content"


def test_identical_upload_is_not_indexed_again(tmp_path, monkeypatch):
    monkeypatch.setitem(app.config, "DOCUMENTS_DIRECTORY", tmp_path)
    (tmp_path / "existing.txt").write_bytes(b"same content")
    client = app.test_client()

    with patch("app.index_file") as index_file:
        response = client.post(
            "/upload",
            data={"document": (io.BytesIO(b"same content"), "copy.txt")},
            content_type="multipart/form-data",
        )

    assert b"This file is already uploaded and indexed." in response.data
    index_file.assert_not_called()
    assert not (tmp_path / "copy.txt").exists()


def test_upload_rejects_unsupported_file_type(tmp_path, monkeypatch):
    monkeypatch.setitem(app.config, "DOCUMENTS_DIRECTORY", tmp_path)
    client = app.test_client()

    response = client.post(
        "/upload",
        data={"document": (io.BytesIO(b"not a document"), "notes.docx")},
        content_type="multipart/form-data",
    )

    assert b"Only PDF and TXT files can be uploaded." in response.data
    assert list(tmp_path.iterdir()) == []


def test_upload_rejects_files_larger_than_ten_megabytes(tmp_path, monkeypatch):
    monkeypatch.setitem(app.config, "DOCUMENTS_DIRECTORY", tmp_path)
    client = app.test_client()

    with patch("app.index_file") as index_file:
        response = client.post(
            "/upload",
            data={"document": (io.BytesIO(b"x" * (10 * 1024 * 1024 + 1)), "large.txt")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 413
    assert b"File uploads must be 10 MB or smaller." in response.data
    index_file.assert_not_called()


def test_upload_hides_indexing_error_details(tmp_path, monkeypatch):
    monkeypatch.setitem(app.config, "DOCUMENTS_DIRECTORY", tmp_path)
    client = app.test_client()

    with patch("app.index_file", side_effect=RuntimeError("sensitive indexing detail")):
        response = client.post(
            "/upload",
            data={"document": (io.BytesIO(b"upload content"), "report.txt")},
            content_type="multipart/form-data",
        )

    assert b"Unable to index that file. Please try again." in response.data
    assert b"sensitive indexing detail" not in response.data
    assert not (tmp_path / "report.txt").exists()
