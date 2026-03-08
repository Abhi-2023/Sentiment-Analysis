import numpy as np
from src.embeddings import train_tfidf_weighted_embeddings
import pickle
from gensim.models import Word2Vec

model = pickle.load(open("models/sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
wv_model = Word2Vec.load("models/word2vec.model")

def predict_sentiment(review):
    review = review.lower()
    
    vector = train_tfidf_weighted_embeddings(review, wv_model=wv_model, tfidf=tfidf)
    vector = vector.reshape(1,-1)
    
    pred = model.predict(vector)[0]
    
    if pred == 1:
        return "Postive review"
    else:
        return "Negative review"