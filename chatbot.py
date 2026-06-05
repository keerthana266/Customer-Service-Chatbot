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
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# -----------------------
# LOAD INTENTS
# -----------------------
with open("intents.json", "r") as file:
    intents = json.load(file)


# -----------------------
# PREPARE DATA
# -----------------------
all_patterns = []
tag_map = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern.lower())
        tag_map.append(intent["tag"])


# -----------------------
# TF-IDF MODEL (IMPROVED)
# -----------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 4), lowercase=True, stop_words="english")
X = vectorizer.fit_transform(all_patterns)


# -----------------------
# CHATBOT RESPONSE ENGINE
# -----------------------
def get_response(user_message):
    user_message = user_message.lower()

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)[0]

    # top 2 matches
    top_indices = similarity.argsort()[-2:][::-1]

    best_score = similarity[top_indices[0]]
    second_score = similarity[top_indices[1]]

    # confidence check
    if best_score < 0.18:
        return "Sorry, I didn't understand that."

    # avoid wrong intent confusion
    if best_score - second_score < 0.05:
        return "Can you please rephrase that?"

    tag = tag_map[top_indices[0]]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."