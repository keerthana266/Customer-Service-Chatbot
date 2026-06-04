from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json["message"]

    result = get_response(user_message)

    # If chatbot returns a dictionary
    if isinstance(result, dict):
        return jsonify(result)

    # Default response format
    return jsonify({
        "message": result,
        "confidence": 100
    })

if __name__ == "__main__":
    app.run(debug=True)