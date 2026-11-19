# 📰 True & Fake News Detection – NLP Classification System

> A supervised Natural Language Processing (NLP) system designed to detect and classify news articles as **REAL** or **FAKE** using regex text normalization, NLTK lemmatization, TF-IDF feature extraction, and multi-model machine learning evaluation.

---

## 🌟 Highlights

- **Text Preprocessing**: Robust pipeline incorporating lowercasing, URL/HTML tag stripping, regex tokenization, stop-word removal, and NLTK `WordNetLemmatizer`.
- **Feature Extraction**: Sublinear `TfidfVectorizer` capturing both unigrams and bigrams (`ngram_range=(1, 2)`).
- **Multi-Classifier Comparison**:
  - Passive Aggressive Classifier
  - Multinomial Naive Bayes (MNB)
  - Logistic Regression (L2 Regularized)
  - Linear Support Vector Machine (LinearSVC)
- **Rigorous Evaluation**: Cross-validation (Stratified K-Fold), confusion matrices, precision, recall, and F1-score comparisons.
- **Inference CLI**: Ready-to-use command-line prediction script for analyzing live news excerpts.

---

## 📁 Repository Structure

```
True-Fake-News-Detection/
├── preprocess.py            # Text cleaning, regex stripping & NLTK lemmatization
├── train.py                 # Multi-model training, cross-validation & evaluation
├── predict.py               # Interactive CLI for testing news headlines/articles
├── fake_news_detection.ipynb# Jupyter notebook covering EDA, training & metrics
├── sample_news.csv          # Sample dataset for immediate out-of-the-box training
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kunwar2021/True-Fake-News-Detection.git
cd True-Fake-News-Detection
```

### 2. Set Up Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models
```bash
python train.py --data sample_news.csv
```
This trains all classifiers, performs cross-validation, prints evaluation metrics, and saves the best-performing model to `models/`.

### 5. Predict on Custom Text
```bash
python predict.py "Scientists discover ancient organic molecules on Mars rover expedition"
```

---

## 📊 Methodology & Workflow

1. **Preprocessing & Cleaning**:
   - Strip URLs, HTML markup, digits, and punctuation using regular expressions.
   - Tokenize text and filter out standard English stopwords.
   - Reduce words to their dictionary root using `WordNetLemmatizer`.

2. **Vectorization**:
   - Term Frequency - Inverse Document Frequency (`TF-IDF`) vectorizer with sublinear term frequency scaling and n-gram ranges up to bigrams.

3. **Classification & Validation**:
   - Evaluated across diverse linear and probabilistic algorithms.
   - Stratified K-Fold cross-validation ensures balanced class distributions across folds.

---

## 📜 License
This project is open-source under the MIT License.
