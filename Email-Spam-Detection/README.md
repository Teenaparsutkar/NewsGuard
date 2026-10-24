@'
# SMS Spam Detection Using Machine Learning

## Overview
This project classifies SMS messages as spam or ham (legitimate messages) using machine learning and Natural Language Processing (NLP).

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit

## Project Workflow
1. Load and inspect the SMS dataset.
2. Remove duplicate messages.
3. Split the data into training and testing sets.
4. Convert text messages into numerical TF-IDF features.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Use the Streamlit dashboard to classify new messages.

## Model Results
- Unique messages after cleaning: 5,171
- Training messages: 4,136
- Testing messages: 1,035
- Accuracy: 95.27%
- Spam precision: 100%
- Spam recall: 62.60%

**Limitation:** The model missed some spam messages during testing. A message classified as ham is not guaranteed to be safe.

## How to Run

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt