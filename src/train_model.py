import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from src.preprocessing import text_preprocessing
from src.embeddings import (
    train_word_2_vec_model,
    train_tfidf,
    train_tfidf_weighted_embeddings,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_model():

    print("Loading dataset...")

    df = pd.read_csv("data/IMDB Dataset.csv")

    encoder = LabelEncoder()
    df["sentiment"] = encoder.fit_transform(df["sentiment"])

    df.drop_duplicates(inplace=True)

    print("Preprocessing text...")

    corpus = text_preprocessing(df, "review")

    print("Training Word2Vec...")

    wv_model = train_word_2_vec_model(corpus=corpus)

    print("Training TF-IDF...")

    tfIdf, processed_corpus = train_tfidf(corpus=corpus)

    print("Creating embeddings...")

    X = np.array([
        train_tfidf_weighted_embeddings(
            sentence=sentence,
            wv_model=wv_model,
            tfidf=tfIdf
        )
        for sentence in processed_corpus
    ])

    y = df["sentiment"]

    print("Scaling features...")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("Training Logistic Regression...")

    model = LogisticRegression(C=2, penalty="l2", solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    print("Saving models...")

    os.makedirs(MODEL_DIR, exist_ok=True)

    pickle.dump(model, open(os.path.join(MODEL_DIR, "sentiment_model.pkl"), "wb"))
    pickle.dump(tfIdf, open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb"))

    wv_model.save(os.path.join(MODEL_DIR, "word2vec.model"))

    print("✅ Models successfully saved in /models folder")


if __name__ == "__main__":
    train_model()