
import pandas as pd
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
data_path = "dataset/SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    quoting=csv.QUOTE_NONE
)

print("Original dataset size:", len(df))

# 2. Remove duplicate records
df = df.drop_duplicates().copy()

# Convert labels: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Separate messages and labels
X = df["message"]
y = df["label"]

print("Dataset size after cleaning:", len(df))
print("Ham messages:", (y == 0).sum())
print("Spam messages:", (y == 1).sum())

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# 4. Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF feature extraction completed.")

# 5. Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# 6. Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print("Rows = Actual labels; Columns = Predicted labels")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# 7. Test the model with new messages
sample_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hi, are we still meeting tomorrow?"
]

sample_tfidf = vectorizer.transform(sample_messages)
sample_predictions = model.predict(sample_tfidf)

print("\n========== SAMPLE PREDICTIONS ==========")

for message, prediction in zip(sample_messages, sample_predictions):
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"\nMessage: {message}")
    print(f"Prediction: {label}")
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")