# 🎬 Movie Review Sentiment Analysis

A full-stack **Natural Language Processing (NLP)** project that predicts the sentiment of movie reviews using **Word2Vec embeddings + TF-IDF weighted sentence vectors + Logistic Regression**, deployed with **FastAPI** and a simple **web interface**.

Users can enter a movie review and instantly receive the predicted sentiment (**Positive / Negative**).

---

## 🚀 Features

- NLP text preprocessing pipeline
- Word2Vec word embeddings
- TF-IDF weighted sentence embeddings
- Logistic Regression classification model
- FastAPI backend
- HTML user interface
- Model persistence (saved models)
- Ready for cloud deployment

---

## 🧠 Machine Learning Pipeline

# 🎬 Movie Review Sentiment Analysis

A full-stack **Natural Language Processing (NLP)** project that predicts the sentiment of movie reviews using **Word2Vec embeddings + TF-IDF weighted sentence vectors + Logistic Regression**, deployed with **FastAPI** and a simple **web interface**.

Users can enter a movie review and instantly receive the predicted sentiment (**Positive / Negative**).

---

## 🚀 Features

- NLP text preprocessing pipeline
- Word2Vec word embeddings
- TF-IDF weighted sentence embeddings
- Logistic Regression classification model
- FastAPI backend
- HTML user interface
- Model persistence (saved models)
- Ready for cloud deployment

---

## 🧠 Machine Learning Pipeline
User Review
↓
Text Preprocessing
↓
Tokenization + Stopword Removal + Lemmatization
↓
Word2Vec Embedding
↓
TF-IDF Weighting
↓
Sentence Vector
↓
Logistic Regression Model
↓
Sentiment Prediction


---

## 📊 Dataset

Dataset used:

**IMDB Movie Review Dataset**

Contains **50,000 movie reviews** labeled as:

- Positive
- Negative

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis
```

Create Virtual Environment
```
python -m venv venv
```

Activate Environment

Windows
```
python -m venv venv
```

Install Dependencies

```
pip install -r requirements.txt
```

Train the Model

Run the training pipeline:

```
python -m src.train_model
```

Run The Application

```
uvicorn main:app --reload
```

Open Your Browser With Below URL

```
http://127.0.0.1:8000
```

Enter a movie review and see the predicted sentiment.