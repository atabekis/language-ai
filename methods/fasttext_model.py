import os
import fasttext
from config import __PROJECT_PATH__
from sklearn.base import BaseEstimator, TransformerMixin


class FastTextModel(BaseEstimator, TransformerMixin):
    """Represents the FastText model to be used in a Pipeline
    :param dim: int, default 300;
        The dimensionality of the vectorizer.
    :param epoch: int, default 10;
        Number of epochs to train for.
    :param lr: float, default 1.0,
        learning rate of the model.
    :param word_ngrams: int default 2;
        Word n-grams to be trained on
    """

    def __init__(self, dim=300, epoch=10, lr=0.8, word_ngrams=2):
        self.model = None
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams

    def fit(self, X=None, y=None):
        # Train FastText model
        self.model = fasttext.train_supervised(os.path.join(__PROJECT_PATH__, 'data', 'fasttext_train.txt'),
                                               dim=self.dim, epoch=self.epoch, lr=self.lr, wordNgrams=self.word_ngrams,
                                               thread=1)
        return self

    def transform(self, X):
        return [self.model.get_sentence_vector(text) for text in X]

    def predict(self, X):
        pred = [self.model.predict(text)[0][0].split("__")[-1] for text in X]
        return [int(prediction) for prediction in pred]

    def get_word_weights(self, N):
        word_embeddings = self.model.get_input_matrix()

        words = self.model.get_words()
        word_importance = {word: sum(abs(weight)) for word, weight in zip(words, word_embeddings)}

        sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_words[:N])

