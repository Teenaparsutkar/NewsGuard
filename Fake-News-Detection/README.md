# 📰 Fake News Detection System

An AI-powered Natural Language Processing (NLP) application that classifies news articles as **Fake** or **Real** using Machine Learning.

## 📌 Project Overview

The Fake News Detection System analyzes the textual content of a news article and predicts whether it is likely to be fake or real.

The system uses:

- Text preprocessing
- TF-IDF feature extraction
- Multiple Machine Learning algorithms
- Model evaluation
- Flask web application

The best-performing model, **Linear SVM**, achieved an accuracy of **99.47%** on the test dataset.

> ⚠️ Disclaimer: This system performs machine-learning-based classification. It does not independently verify news against live sources or guarantee factual accuracy.

---

## 🎯 Objectives

- Detect potentially fake news using NLP and Machine Learning.
- Clean and preprocess raw news articles.
- Convert textual data into numerical TF-IDF features.
- Compare multiple Machine Learning algorithms.
- Evaluate model performance using standard classification metrics.
- Provide a simple web interface for users to analyze news.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning & NLP
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

### Data Processing
- Pandas
- NumPy
- Regular Expressions

### Visualization
- Matplotlib
- Seaborn

### Web Development
- Flask
- HTML
- CSS

### Model Serialization
- Joblib

---

## 📂 Dataset

The project uses a labeled news dataset containing separate files for fake and real news:

- `Fake.csv`
- `True.csv`

The original dataset contains:

- 23,481 fake news records
- 21,417 real news records

After preprocessing and removal of invalid/duplicate records, the final dataset contains:

**38,826 articles**

### Final Label Distribution

| Label | Articles |
|---|---:|
| Fake | 17,900 |
| Real | 20,926 |

---

## 🔄 System Workflow

```text
Raw News Dataset
       ↓
Data Cleaning & Preprocessing
       ↓
Text Normalization
       ↓
TF-IDF Feature Extraction
       ↓
Train/Test Split
       ↓
Machine Learning Models
       ↓
Model Comparison
       ↓
Best Model: Linear SVM
       ↓
News Prediction
       ↓
Fake / Real Result