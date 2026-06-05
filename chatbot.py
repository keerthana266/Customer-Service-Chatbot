import json
import random
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# NLTK SAFE SETUP FOR RENDER
# -------------------------

NLTK_DIR = "/opt/render/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)


def ensure_nltk():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', download_dir=NLTK_DIR)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DIR)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', download_dir=NLTK_DIR)


ensure_nltk()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()