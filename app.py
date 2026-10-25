from flask import Flask, render_template, request

from src.rag import answer_question


app = Flask(__name__)


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

    return render_template("index.html", question=question, answer=answer, error=error)


if __name__ == "__main__":
    app.run(debug=True)
