import os
import joblib
import numpy as np
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODELS_DIR = os.path.join("models")


class TFIDFExtractor:
    """
    TF-IDF Vectorizer wrapper with configurable max_features and n-gram range.
    """
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True
        )

    def fit_transform(self, texts: list):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list):
        return self.vectorizer.transform(texts)

    def save(self, filepath: str = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"[Features] Saved TF-IDF vectorizer to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")):
        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        print(f"[Features] Loaded TF-IDF vectorizer from '{filepath}'.")
        return instance


class SequentialExtractor:
    """
    Keras Tokenizer and padded sequence extractor for LSTM / Bi-LSTM path.
    """
    def __init__(self, num_words: int = 10000, max_len: int = 150):
        self.num_words = num_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.num_words, oov_token="<OOV>")

    def fit_transform(self, texts: list) -> np.ndarray:
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def transform(self, texts: list) -> np.ndarray:
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def save(self, filepath: str = os.path.join(MODELS_DIR, "lstm_tokenizer.joblib")):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.tokenizer, filepath)
        print(f"[Features] Saved LSTM tokenizer to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str = os.path.join(MODELS_DIR, "lstm_tokenizer.joblib"), max_len: int = 150):
        instance = cls(max_len=max_len)
        instance.tokenizer = joblib.load(filepath)
        print(f"[Features] Loaded LSTM tokenizer from '{filepath}'.")
        return instance
