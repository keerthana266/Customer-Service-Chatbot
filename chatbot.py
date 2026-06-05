import json
import random
import nltk
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# NLTK SETUP
# -----------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


# -----------------------
# LOAD INTENTS
# -----------------------
with open("intents.json", "r") as file:
    intents = json.load(file)


# -----------------------
# PREPARE DATA
# -----------------------
patterns = []
tags = []

for intent in intents["intents"]:
    for p in intent["patterns"]:
        patterns.append(p.lower())
        tags.append(intent["tag"])


# -----------------------
# IMPROVED TF-IDF MODEL
# -----------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(patterns)


# -----------------------
# CHATBOT RESPONSE ENGINE
# -----------------------
def get_response(user_message):
    user_message = user_message.lower()

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)[0]

    best_index = np.argmax(similarity)
    best_score = similarity[best_index]

    # real confidence score (0–100%)
    confidence = round(best_score * 100, 2)

    # stricter threshold (better accuracy)
    if best_score < 0.18:
        return {
            "message": "Sorry, I didn't understand that. Can you rephrase?",
            "confidence": confidence
        }

    tag = tags[best_index]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return {
                "message": random.choice(intent["responses"]),
                "confidence": confidence
            }

    return {
        "message": "Sorry, I didn't understand that.",
        "confidence": confidence
    }