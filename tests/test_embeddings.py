from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.embeddings import EMBEDDING_MODEL, get_embeddings


def test_get_embeddings_uses_openai_embedding_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
    )
    client = Mock()
    client.embeddings.create.return_value = response

    with patch("src.embeddings.OpenAI", return_value=client) as openai_client:
        embeddings = get_embeddings(["first", "second"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    openai_client.assert_called_once_with(api_key="test-key")
    client.embeddings.create.assert_called_once_with(
        model=EMBEDDING_MODEL, input=["first", "second"]
    )


def test_get_embeddings_skips_api_call_for_empty_input():
    with patch("src.embeddings.OpenAI") as openai_client:
        assert get_embeddings([]) == []

    openai_client.assert_not_called()


def test_get_embeddings_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("src.embeddings.load_dotenv"):
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            get_embeddings(["text"])
