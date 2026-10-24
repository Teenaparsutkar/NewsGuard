
from sklearn.feature_extraction.text import TfidfVectorizer

messages = [
    "win free prize",
    "win cash prize",
    "meeting scheduled for tomorrow"
]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(messages)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF matrix:")
print(matrix.toarray().round(3))
