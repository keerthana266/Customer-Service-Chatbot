from flask import Flask, render_template, request, jsonify
from chatbot import get_response
import os

app = Flask(__name__)


# -------------------------
# Home Route
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Chat API Route
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        # safer extraction (prevents crashes)
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({
                "message": "Please enter a message.",
                "confidence": 0
            })

        result = get_response(user_message)

        # If chatbot returns a dictionary
        if isinstance(result, dict):
            return jsonify(result)

        # Default response format
        return jsonify({
            "message": result,
            "confidence": 100
        })

    except Exception as e:
        # prevents Render crash logs from breaking app
        return jsonify({
            "message": "Server error occurred. Please try again.",
            "error": str(e),
            "confidence": 0
        })


# -------------------------
# Run App (Render Safe)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)