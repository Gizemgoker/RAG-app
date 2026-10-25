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
