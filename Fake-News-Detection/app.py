from flask import Flask, render_template, request
import joblib
import re
import numpy as np


app = Flask(__name__)


# -------------------------------------------------
# Load trained ML model and TF-IDF vectorizer
# -------------------------------------------------

model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# -------------------------------------------------
# Text cleaning
# -------------------------------------------------

def clean_text(text):

    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------
# Prediction function
# -------------------------------------------------

def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    text_tfidf = tfidf.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]

    decision_score = model.decision_function(text_tfidf)[0]

    # Confidence-like score
    confidence = 1 / (1 + np.exp(-abs(decision_score)))
    confidence = confidence * 100

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence


# -------------------------------------------------
# Home page
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    news_text = ""

    if request.method == "POST":

        news_text = request.form.get("news_text", "").strip()

        if news_text:

            result, confidence = predict_news(news_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        news_text=news_text
    )


# -------------------------------------------------
# Run Flask application
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)