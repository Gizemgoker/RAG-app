import hashlib
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.document_loader import SUPPORTED_EXTENSIONS
from src.indexer import index_file
from src.rag import answer_question


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["DOCUMENTS_DIRECTORY"] = Path("documents")


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Please enter a question."
        else:
            try:
                answer = answer_question(question)
            except Exception:
                app.logger.exception("Unable to answer question")
                error = "Unable to answer your question right now. Please try again."

    return render_template("index.html", question=question, answer=answer, error=error, success=None)


@app.post("/upload")
def upload_document():
    uploaded_file = request.files.get("document")
    if uploaded_file is None or not uploaded_file.filename:
        return _upload_response(error="Choose a PDF or TXT file to upload.")

    filename = secure_filename(uploaded_file.filename)
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return _upload_response(error="Only PDF and TXT files can be uploaded.")

    content = uploaded_file.read()
    if not content:
        return _upload_response(error="The uploaded file is empty.")

    documents_directory = Path(app.config["DOCUMENTS_DIRECTORY"])
    documents_directory.mkdir(parents=True, exist_ok=True)
    if _contains_identical_file(content, documents_directory):
        return _upload_response(success="This file is already uploaded and indexed.")

    destination = _available_destination(documents_directory, filename)
    destination.write_bytes(content)
    try:
        chunk_count = index_file(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        app.logger.exception("Unable to index uploaded document")
        return _upload_response(error="Unable to index that file. Please try again.")

    chunk_label = "chunk" if chunk_count == 1 else "chunks"
    return _upload_response(success=f"Uploaded and indexed {chunk_count} {chunk_label}.")


@app.errorhandler(413)
def file_too_large(_error):
    return _upload_response(error="File uploads must be 10 MB or smaller."), 413


def _upload_response(success=None, error=None):
    return render_template("index.html", question="", answer=None, error=error, success=success)


def _contains_identical_file(content: bytes, directory: Path) -> bool:
    content_hash = hashlib.sha256(content).digest()
    for path in directory.iterdir():
        if path.is_file() and path.name != ".gitkeep" and _file_hash(path) == content_hash:
            return True
    return False


def _file_hash(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as document:
        for block in iter(lambda: document.read(65536), b""):
            digest.update(block)
    return digest.digest()


def _available_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{Path(filename).stem}_{suffix}{Path(filename).suffix}"
        suffix += 1
    return candidate


if __name__ == "__main__":
    app.run(debug=True)
