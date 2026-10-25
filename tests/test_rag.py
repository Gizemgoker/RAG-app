from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.rag import ANSWER_MODEL, NO_CONTEXT_ANSWER, answer_question


def test_answer_question_uses_retrieved_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chunks = [
        {"text": "Paris is the capital of France.", "source": "facts.txt", "page": None, "chunk_id": 0}
    ]
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text="Paris.")

    with (
        patch("src.rag.retrieve_chunks", return_value=chunks) as retrieve_chunks,
        patch("src.rag.OpenAI", return_value=client) as openai_client,
    ):
        answer = answer_question("What is the capital of France?")

    assert answer == "Paris."
    retrieve_chunks.assert_called_once_with("What is the capital of France?")
    openai_client.assert_called_once_with(api_key="test-key")
    client.responses.create.assert_called_once_with(
        model=ANSWER_MODEL,
        instructions=(
            "Answer only using the provided document context. "
            "If the context does not contain the answer, say that you do not know."
        ),
        input=(
            "Document context:\nSource: facts.txt | Page: None | Chunk: 0\n"
            "Paris is the capital of France.\n\nQuestion: What is the capital of France?"
        ),
    )


def test_answer_question_skips_retrieval_for_blank_question():
    with patch("src.rag.retrieve_chunks") as retrieve_chunks:
        answer = answer_question("   ")

    assert answer == "Please enter a question."
    retrieve_chunks.assert_not_called()


def test_answer_question_returns_fallback_when_retrieval_has_no_chunks():
    with (
        patch("src.rag.retrieve_chunks", return_value=[]),
        patch("src.rag.OpenAI") as openai_client,
    ):
        answer = answer_question("What is the answer?")

    assert answer == NO_CONTEXT_ANSWER
    openai_client.assert_not_called()
