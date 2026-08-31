import pandas as pd
import re


# -------------------------------------------------
# 1. Load datasets
# -------------------------------------------------

fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")

print("Fake news records:", len(fake))
print("Real news records:", len(real))


# -------------------------------------------------
# 2. Add labels
# -------------------------------------------------

fake["label"] = 0
real["label"] = 1


# -------------------------------------------------
# 3. Combine datasets
# -------------------------------------------------

df = pd.concat([fake, real], ignore_index=True)

print("\nCombined dataset shape:", df.shape)


# -------------------------------------------------
# 4. Combine title and article text
# -------------------------------------------------

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")


# -------------------------------------------------
# 5. Text cleaning function
# -------------------------------------------------

def clean_text(text):
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------
# 6. Apply text cleaning
# -------------------------------------------------

df["clean_content"] = df["content"].apply(clean_text)


# -------------------------------------------------
# 7. Remove empty and duplicate records
# -------------------------------------------------

df = df[df["clean_content"].str.len() > 0]

df = df.drop_duplicates(subset=["clean_content"])


# -------------------------------------------------
# 8. Keep required columns
# -------------------------------------------------

df = df[["clean_content", "label"]]


# -------------------------------------------------
# 9. Shuffle dataset
# -------------------------------------------------

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# -------------------------------------------------
# 10. Display information
# -------------------------------------------------

print("\nFinal dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSample cleaned article:")
print(df["clean_content"].iloc[0][:500])


# -------------------------------------------------
# 11. Save processed dataset
# -------------------------------------------------

df.to_csv("data/processed_news.csv", index=False)

print("\nProcessed dataset saved successfully!")