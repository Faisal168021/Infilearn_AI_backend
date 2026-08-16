"""
train_classifier.py
--------------------
Standalone: dataset/dataset.csv -> TF-IDF + Naive Bayes -> save .pkl
Run: python train_classifier.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from app.config import settings
from app.services.text_cleaner import tokenize_for_ml


def load_dataset():
    csv_path = os.path.join(settings.DATASET_DIR, "dataset.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "category"])
    return df


def preprocess_for_tfidf(text: str) -> str:
    tokens = tokenize_for_ml(text)
    return " ".join(tokens)


def train():
    print("Loading dataset...")
    df = load_dataset()
    print(f"   {len(df)} samples, {df['category'].nunique()} categories")

    df["clean_text"] = df["text"].apply(preprocess_for_tfidf)
    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(settings.ML_MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(settings.ML_MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(settings.ML_MODELS_DIR, "naive_bayes_model.pkl"))
    print(f"\nModel saved to: {settings.ML_MODELS_DIR}")


if __name__ == "__main__":
    train()
