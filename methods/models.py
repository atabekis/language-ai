import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from tqdm import tqdm


class Word2VecModel(BaseEstimator, TransformerMixin):
    """This is the WordEmbedding model."""

    def __init__(self, size=100, window=5, min_count=1, workers=3):
        self.size = size
        self.window = window,
        self.min_count = min_count,
        self.workers = workers

        self.model = None

    @staticmethod
    def _tokenize_sentences(sentences):
        tokenized_sentences = []
        for sentence in tqdm(sentences, desc="Tokenizing sentences", unit="sentence"):
            tokenized_sentences.append(word_tokenize(sentence))
        return tokenized_sentences

    def fit(self, X, y=None):

        X_tokenized = self._tokenize_sentences(X)

        self.model = Word2Vec(
            sentences=X_tokenized,
            vector_size=self.size,
            window=int(self.window),
            min_count=self.min_count,
            workers=self.workers
        )

        return self

    def transform(self, X):
        X_tokenized = self._tokenize_sentences(X)

        emb = [np.mean(
            [self.model.wv[word] for word in sentence if word in self.model.wv]
            or [np.zeros(self.size)], axis=0) for sentence in X_tokenized]

        return np.vstack(emb)
