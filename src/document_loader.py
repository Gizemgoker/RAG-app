"""Load local documents and split their text into overlapping chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

logger = logging.getLogger(__name__)


def validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """Raise ValueError when chunking parameters cannot make progress."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be at least 0 and less than chunk_size")


def chunk_text(
    text: str,
    source: str,
    page: int | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_id: int = 0,
) -> list[dict[str, Any]]:
    """Split one text unit into non-empty character chunks."""
    validate_chunk_parameters(chunk_size, chunk_overlap)

    if not text or not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(
                {
                    "text": chunk,
                    "source": source,
                    "page": page,
                    "chunk_id": start_chunk_id + len(chunks),
                }
            )
        if start + chunk_size >= len(text):
            break
        start += step

    return chunks


def load_documents(
    documents_dir: str | Path = "documents",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Load supported files in filename order and return their text chunks.

    Unreadable files are skipped and reported through this module's logger.
    """
    validate_chunk_parameters(chunk_size, chunk_overlap)
    directory = Path(documents_dir)

    if not directory.exists():
        logger.warning("Documents directory does not exist: %s", directory)
        return []

    chunks: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda item: item.name.lower(),
    ):
        chunks.extend(_load_file(path, chunk_size, chunk_overlap))

    return chunks


def _load_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".txt":
            text_units = [(path.read_text(encoding="utf-8"), None)]
        else:
            reader = PdfReader(path)
            text_units = [(page.extract_text() or "", number) for number, page in enumerate(reader.pages, start=1)]
    except (OSError, PdfReadError, ValueError) as error:
        logger.warning("Unable to read document %s: %s", path.name, error)
        return []

    file_chunks: list[dict[str, Any]] = []
    for text, page_number in text_units:
        file_chunks.extend(
            chunk_text(
                text,
                source=path.name,
                page=page_number,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                start_chunk_id=len(file_chunks),
            )
        )
    return file_chunks
