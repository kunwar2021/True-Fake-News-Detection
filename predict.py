"""
Inference Script for Real-Time News Classification
Classifies user-supplied news headline/article as REAL or FAKE.
"""

import os
import sys
import joblib
from preprocess import clean_text

def predict_news(text: str, model_path: str = "models/best_classifier.joblib", vectorizer_path: str = "models/tfidf_vectorizer.joblib"):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("❌ Trained model or vectorizer not found. Run 'python train.py' first.")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = max(probs) * 100
    elif hasattr(model, "decision_function"):
        df = abs(model.decision_function(features)[0])
        confidence = min(99.0, 50.0 + df * 10.0)

    print("\n" + "=" * 50)
    print(f"Prediction: {'🟢 REAL' if prediction == 'REAL' else '🔴 FAKE'}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2f}%")
    print("=" * 50)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a news headline or excerpt: ")

    if query.strip():
        predict_news(query)
    else:
        print("Please provide non-empty text to analyze.")
