"""
Text Preprocessing Pipeline for True & Fake News Detection
Handles regex cleaning, lowercasing, tokenization, stop-word removal,
and NLTK WordNet lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data packages are downloaded
for resource in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()


def clean_text(text: str) -> str:
    """
    Cleans raw article text through:
    1. Lowercasing
    2. Removing URLs, HTML tags, numbers, and special characters
    3. Tokenizing words
    4. Removing English stop words
    5. Lemmatizing words using NLTK WordNetLemmatizer
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters, punctuation, and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize words
    words = word_tokenize(text) if text else []

    # Filter stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_tokens)
