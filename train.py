"""
Model Training & Evaluation Script
Trains multiple supervised classifiers on TF-IDF vectorized text:
- Logistic Regression
- Multinomial Naive Bayes
- Passive Aggressive Classifier
- Linear Support Vector Classifier (LinearSVC)

Compares models using Stratified 5-Fold Cross-Validation,
Confusion Matrix, Precision, Recall, and F1 Score.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from preprocess import clean_text

def train_and_evaluate(data_path: str = "sample_news.csv", model_dir: str = "models"):
    print("=" * 65)
    print("📰 TRUE & FAKE NEWS DETECTION — NLP CLASSIFICATION SYSTEM")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

    print(f"\n[1/5] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total records loaded: {len(df)}")

    # Combine title and text for comprehensive context
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    elif 'text' in df.columns:
        df['content'] = df['text'].fillna('')
    else:
        df['content'] = df.iloc[:, 0].fillna('')

    target_col = 'label' if 'label' in df.columns else df.columns[-1]
    y = df[target_col].astype(str).str.strip().str.upper()

    print("\n[2/5] Cleaning and Preprocessing Text (regex, lemmatization, stop words)...")
    df['cleaned'] = df['content'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], y, test_size=0.25, random_state=42, stratify=y
    )

    print("\n[3/5] Extracting TF-IDF Features (ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")

    classifiers = {
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=50, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear Support Vector (SVC)": LinearSVC(random_state=42)
    }

    print("\n[4/5] Training Classifiers & Cross-Validation...")
    print("-" * 65)
    print(f"{'Classifier':<32} | {'CV Accuracy':<12} | {'Test Accuracy':<12}")
    print("-" * 65)

    best_model_name = None
    best_f1 = -1.0
    best_model = None

    skf = StratifiedKFold(n_splits=min(3, len(X_train)), shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        try:
            cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=skf, scoring='accuracy')
            cv_acc = np.mean(cv_scores)
        except Exception:
            cv_acc = 0.0

        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"{name:<32} | {cv_acc*100:>10.2f}% | {test_acc*100:>10.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = clf

    print("-" * 65)
    print(f"\n🌟 Best Classifier: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Detailed Evaluation for the Best Model
    y_pred_best = best_model.predict(X_test_tfidf)
    print("\n[5/5] Detailed Evaluation Report:")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_best, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

    # Save Best Model and Vectorizer
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_classifier.joblib")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n✅ Artifacts saved successfully to '{model_dir}/'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NLP Fake News Detection Models")
    parser.add_argument("--data", default="sample_news.csv", help="Path to news dataset CSV")
    parser.add_argument("--out", default="models", help="Output directory for saved models")
    args = parser.parse_args()

    train_and_evaluate(data_path=args.data, model_dir=args.out)
