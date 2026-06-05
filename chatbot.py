import json
import random
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# NLTK SETUP (Render safe)
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
        all_patterns.append(pattern)
        tag_map.append(intent["tag"])


# -----------------------
# TF-IDF MODEL
# -----------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(all_patterns)


# -----------------------
# GET RESPONSE FUNCTION
# -----------------------
def get_response(user_message):
    user_message = user_message.lower()

    user_vec = vectorizer.transform([user_message])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    # confidence threshold
    if score < 0.15:
        return "Sorry, I didn't understand that."

    tag = tag_map[index]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."