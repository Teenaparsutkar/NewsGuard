import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# -------------------------------------------------
# 1. Load processed dataset
# -------------------------------------------------

df = pd.read_csv("data/processed_news.csv")

X = df["clean_content"]
y = df["label"]


# -------------------------------------------------
# 2. Create the same train/test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------------------------------------
# 3. Load trained model and TF-IDF vectorizer
# -------------------------------------------------

model = joblib.load("models/fake_news_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# -------------------------------------------------
# 4. Transform test data using TF-IDF
# -------------------------------------------------

X_test_tfidf = tfidf.transform(X_test)


# -------------------------------------------------
# 5. Make predictions
# -------------------------------------------------

predictions = model.predict(X_test_tfidf)


# -------------------------------------------------
# 6. Calculate metrics
# -------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# -------------------------------------------------
# 7. Classification report
# -------------------------------------------------

report = classification_report(
    y_test,
    predictions,
    target_names=["Fake", "Real"]
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)


# -------------------------------------------------
# 8. Save classification report
# -------------------------------------------------

with open("results/classification_report.txt", "w") as file:
    file.write("Fake News Detection - Classification Report\n\n")
    file.write(report)

    file.write("\nOverall Metrics\n")
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")


# -------------------------------------------------
# 9. Generate confusion matrix
# -------------------------------------------------

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Fake News Detection - Confusion Matrix")

plt.tight_layout()

plt.savefig("results/confusion_matrix.png", dpi=300)

plt.show()


print("\nConfusion matrix saved successfully!")
print("Classification report saved successfully!")