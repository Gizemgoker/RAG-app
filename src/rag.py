"""Generate grounded answers from retrieved document chunks."""

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.retriever import retrieve_chunks


ANSWER_MODEL = "gpt-4.1-mini"
NO_CONTEXT_ANSWER = "I don't know based on the provided documents."


def answer_question(query: str) -> str:
    """Answer a question using only the context returned by retrieval."""
    if not query.strip():
        return "Please enter a question."

    chunks = retrieve_chunks(query)
    if not chunks:
        return NO_CONTEXT_ANSWER

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=ANSWER_MODEL,
        instructions=(
            "Answer only using the provided document context. "
            "If the context does not contain the answer, say that you do not know."
        ),
        input=f"Document context:\n{_format_context(chunks)}\n\nQuestion: {query}",
    )
    return response.output_text


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format chunk text with source metadata for the model prompt."""
    return "\n\n".join(
        f"Source: {chunk['source']} | Page: {chunk['page']} | Chunk: {chunk['chunk_id']}\n"
        f"{chunk['text']}"
        for chunk in chunks
    )
