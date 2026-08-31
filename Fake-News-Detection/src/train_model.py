import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------------------------------
# 1. Load processed dataset
# -------------------------------------------------

df = pd.read_csv("data/processed_news.csv")

print("Dataset shape:", df.shape)


# -------------------------------------------------
# 2. Separate features and target
# -------------------------------------------------

X = df["clean_content"]
y = df["label"]


# -------------------------------------------------
# 3. Split dataset
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -------------------------------------------------
# 4. Convert text into TF-IDF features
# -------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# -------------------------------------------------
# 5. Define models
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC()
}


# -------------------------------------------------
# 6. Train and evaluate models
# -------------------------------------------------

results = {}

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# -------------------------------------------------
# 7. Display model comparison
# -------------------------------------------------

print("\n================ MODEL COMPARISON ================")

for name, metrics in results.items():

    print(
        f"{name}: "
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1_score']:.4f}"
    )


# -------------------------------------------------
# 8. Select best model based on F1 score
# -------------------------------------------------

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# -------------------------------------------------
# 9. Save model and TF-IDF vectorizer
# -------------------------------------------------

joblib.dump(
    best_model,
    "models/fake_news_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully!")
print("TF-IDF vectorizer saved successfully!")