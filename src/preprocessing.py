import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_review(review):

    review = str(review)

    # Remove HTML
    review = re.sub(r"<.*?>", " ", review)

    # Remove URLs
    review = re.sub(r"https?://\S+|www\.\S+", "", review)

    # Remove mentions
    review = re.sub(r"@\S+", "", review)

    # Remove special characters
    review = re.sub(r"[^a-zA-Z0-9]", " ", review)

    # Lowercase
    review = review.lower()

    # Remove extra spaces
    review = re.sub(r"\s+", " ", review).strip()

    # Tokenize
    tokens = word_tokenize(review)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def text_preprocessing(df, column):

    corpus = []

    for review in df[column]:

        tokens = preprocess_review(review)

        corpus.append(tokens)

    return corpus