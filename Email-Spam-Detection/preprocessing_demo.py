
import re

message = "WIN a FREE prize now!"

# Convert text to lowercase
cleaned_text = message.lower()

# Extract individual words
tokens = re.findall(r"\b[a-z]+\b", cleaned_text)

print("Original message:", message)
print("Cleaned text:", cleaned_text)
print("Tokens:", tokens)

from sklearn.feature_extraction.text import CountVectorizer

messages = [
    "win free prize",
    "win cash prize"
]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(messages)

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

print("\nDocument-term matrix:")
print(matrix.toarray())
