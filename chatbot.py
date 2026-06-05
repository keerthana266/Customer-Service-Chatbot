import json
import random
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# NLTK SETUP
# -----------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# -----------------------
# TEXT PREPROCESSING
# -----------------------
def preprocess(text):
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


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
        cleaned = preprocess(p)
        patterns.append(cleaned)
        tags.append(intent["tag"])


# -----------------------
# TF-IDF MODEL (IMPROVED)
# -----------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 4),
    lowercase=True,
    stop_words="english",
    sublinear_tf=True
)

X = vectorizer.fit_transform(patterns)


# -----------------------
# CHATBOT ENGINE
# -----------------------
def get_response(user_message):
    user_message = preprocess(user_message)

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)[0]

    best_index = np.argmax(similarity)
    best_score = similarity[best_index]

    confidence = round(best_score * 100, 2)

    # fallback condition
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