import joblib
import re
import numpy as np


# -------------------------------------------------
# 1. Load trained model and TF-IDF vectorizer
# -------------------------------------------------

model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# -------------------------------------------------
# 2. Text cleaning
# -------------------------------------------------

def clean_text(text):

    text = str(text)
    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------
# 3. Prediction
# -------------------------------------------------

def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    text_tfidf = tfidf.transform([cleaned_text])

    prediction = model.predict(text_tfidf)[0]

    # LinearSVC decision score
    decision_score = model.decision_function(text_tfidf)[0]

    # Convert decision score into a confidence-like percentage
    confidence = 1 / (1 + np.exp(-abs(decision_score)))
    confidence = confidence * 100

    if prediction == 0:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence


# -------------------------------------------------
# 4. Main program
# -------------------------------------------------

print("\n===================================")
print("       FAKE NEWS DETECTOR")
print("===================================")

news = input("\nEnter news article or headline: ")

if not news.strip():

    print("\nPlease enter some news text.")

else:

    result, confidence = predict_news(news)

    print("\n-----------------------------------")
    print("Prediction :", result)
    print("Confidence :", f"{confidence:.2f}%")
    print("-----------------------------------")