import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text: str):
    tokens = word_tokenize(text.lower())

    tokens = [
        word for word in tokens
        if word not in string.punctuation
        and word not in stopwords.words("portuguese")
    ]

    return " ".join(tokens)
