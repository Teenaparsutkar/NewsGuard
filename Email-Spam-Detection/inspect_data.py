
import csv
import pandas as pd
from pathlib import Path

# Locate the dataset
data_path = Path("dataset") / "SMSSpamCollection"

# Load the tab-separated file
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    quoting=csv.QUOTE_NONE
)

# Inspect the dataset
print("First five records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nClass distribution in percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

# Verify the dataset
print("\nDataset validation:")
print("Total records:", len(df))
print("Expected records: 5574")
print("Record count correct:", len(df) == 5574)
print("Labels:", df["label"].unique())
print("Duplicate records:", df.duplicated().sum())

# Inspect duplicate records
duplicate_rows = df[df.duplicated(keep=False)]

print("\nTotal rows involved in duplicates:")
print(len(duplicate_rows))

print("\nSample duplicate records:")
print(duplicate_rows.head(10))

print("\nDuplicate counts by label:")
print(duplicate_rows["label"].value_counts())

# Check whether identical messages have conflicting labels
label_counts_per_message = df.groupby("message")["label"].nunique()

conflicting_messages = label_counts_per_message[
    label_counts_per_message > 1
]

print("\nMessages with conflicting labels:")
print(len(conflicting_messages))

if len(conflicting_messages) > 0:
    print("\nExamples of conflicting messages:")
    print(df[df["message"].isin(conflicting_messages.index)].head(10))

# Create a clean dataset by removing exact duplicate rows
df_clean = df.drop_duplicates().copy()

print("\nClean dataset summary:")
print("Original records:", len(df))
print("Records after removing duplicates:", len(df_clean))
print("Duplicate rows removed:", len(df) - len(df_clean))

print("\nClass distribution after cleaning:")
print(df_clean["label"].value_counts())

print("\nClass distribution percentages after cleaning:")
print((df_clean["label"].value_counts(normalize=True) * 100).round(2))
