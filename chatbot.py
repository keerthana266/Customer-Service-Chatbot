import json
import random
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# NLP SETUP
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------------------------
# SAMPLE ORDER DATABASE
# -------------------------
orders = {
    "1001": "Shipped",
    "1002": "Delivered",
    "1003": "Processing",
    "1004": "Out for Delivery"
}

# -------------------------
# SAMPLE PRODUCT DATABASE
# -------------------------
products = {
    "laptop": "₹50,000",
    "mobile": "₹20,000",
    "headphones": "₹2,500",
    "keyboard": "₹1,500",
    "mouse": "₹800"
}

# -------------------------
# LOAD INTENTS
# -------------------------
with open("intents.json", "r") as file:
    intents = json.load(file)

patterns = []
responses = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(intent["responses"])

# -------------------------
# TEXT PREPROCESSING
# -------------------------
def preprocess(text):

    tokens = nltk.word_tokenize(text.lower())

    words = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    return " ".join(words)

# -------------------------
# TRAIN TF-IDF MODEL
# -------------------------
processed_patterns = [preprocess(pattern) for pattern in patterns]

vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(processed_patterns)

# -------------------------
# CHATBOT RESPONSE FUNCTION
# -------------------------
def get_response(user_message):

    user_message_lower = user_message.lower()

    # -------------------------
    # ORDER TRACKING
    # -------------------------
    if "track order" in user_message_lower:

        words = user_message.split()

        for word in words:

            if word.isdigit():

                if word in orders:

                    return {
                        "message": f"📦 Order {word} status: {orders[word]}",
                        "confidence": 100
                    }

                return {
                    "message": "❌ Order ID not found.",
                    "confidence": 100
                }

        return {
            "message": "Please provide an Order ID. Example: Track order 1001",
            "confidence": 100
        }

    # -------------------------
    # PRODUCT LOOKUP
    # -------------------------
    if "price of" in user_message_lower:

        for product in products:

            if product in user_message_lower:

                return {
                    "message": f"💰 {product.title()} price is {products[product]}",
                    "confidence": 100
                }

        return {
            "message": "Product not found.",
            "confidence": 100
        }

    # -------------------------
    # NLP INTENT MATCHING
    # -------------------------
    processed_input = preprocess(user_message)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(
        user_vector,
        pattern_vectors
    )

    best_match_index = similarity.argmax()

    confidence = similarity[0][best_match_index]

    print("User:", user_message)
    print("Confidence:", confidence)
    print("Matched Pattern:", patterns[best_match_index])

    if confidence < 0.75:

        return {
            "message": "Sorry, I didn't understand that. Can you rephrase?",
            "confidence": round(confidence * 100, 2)
        }

    response = random.choice(
        responses[best_match_index]
    )

    return {
        "message": response,
        "confidence": round(confidence * 100, 2)
    }