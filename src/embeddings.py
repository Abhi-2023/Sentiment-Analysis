import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def train_word_2_vec_model(corpus):
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=4,
        sg=1,
        negative=5,
        hs=0,
        min_count=5,
        workers=4,
        epochs=5
    )
    
    return model

def train_tfidf(corpus):
    process_corpus = [" ".join(doc) for doc in corpus]
    
    tfIdf = TfidfVectorizer()
    tfIdf.fit(process_corpus)
    return tfIdf, process_corpus

def train_tfidf_weighted_embeddings(sentence, wv_model, tfidf):
    
    words = sentence.split()
    
    vectors=[]
    weights=[]
    
    for word in words:
        if word in wv_model.wv and word in tfidf.vocabulary_:
            vector = wv_model.wv[word]
            weight = tfidf.idf_[tfidf.vocabulary_[word]]
            
            vectors.append(vector*weight)
            weights.append(weight)
            
    if len(vectors) ==0:
        return np.zeros(wv_model.vector_size)
    
    return np.sum(vectors, axis=0)/np.sum(weights)


        