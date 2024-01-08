import os
import fasttext
from config import __PROJECT_PATH__
from sklearn.base import BaseEstimator, TransformerMixin


class FastTextVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, dim=300, epoch=5, lr=0.1, word_ngrams=2):
        self.model = None
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams

    def fit(self, X, y=None):
        # Train FastText model
        self.model = fasttext.train_supervised(os.path.join(__PROJECT_PATH__, 'data', 'fasttext_train.txt'),
                                               dim=self.dim, epoch=self.epoch, lr=self.lr, wordNgrams=self.word_ngrams,
                                               thread=2)
        return self

    def transform(self, X):
        return [self.model.get_sentence_vector(text) for text in X]

    def predict(self, X):
        pred = [self.model.predict(text)[0][0].split("__")[-1] for text in X]
        return [int(prediction) for prediction in pred]
