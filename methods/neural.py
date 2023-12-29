"""Neural network models: CNN"""
# Python imports
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Neural imports
import keras
import tensorflow as tf  # This is just for tf.string on line 58, can be changed if there's any other option
from keras import Model, Input
from keras.layers import TextVectorization
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense

# We need to pass BaseEstimator and TransformerMixin in order to use the class in a Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class NeuralNetwork(BaseEstimator, TransformerMixin):
    """
    This class represents a neural network to be used in a Sklearn pipeline.

    :param model_type: str, optional, default: 'cnn'
        type of the model. Options: 'cnn'
    :param max_features: int, optional, default: 20000
        Maximum number of features for the model takes into account
    :param embedding_dim: int, optional, default: 128
        Dimensionality of the word embeddings
    :param sequence_length: int, optional, default: 500
        Length of the sequences in the input data
    :param epochs: int, optional, default: 3
        Number of epochs to run the model

    Notes:
    -----
    Some inspiration and structuring from
    https://github.com/cmry/amica/blob/master/neural.py#L131
    """
    def __init__(self,
                 model_type:str = 'cnn',
                 max_features: int = 20000,
                 embedding_dim: int = 128,
                 sequence_length: int = 500,
                 epochs: int = 3) -> None:
        """Initialize the basic control parameters of the neural network"""

        self.model_type = model_type
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None

        self.vectorizer = TextVectorization(
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length)

    def cnn(self, max_features: int, embedding_dim: int) -> keras.Model:
        """Representation of a Convolutional Neural Network.

        Notes:
            Implementation from:
            https://keras.io/examples/nlp/text_classification_from_scratch/
        """
        text_input = Input(shape=(1,), dtype=tf.string, name='text')
        x = self.vectorizer(text_input)
        x = Embedding(max_features + 1, embedding_dim)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation="sigmoid", name="predictions")(x)
        return Model(text_input, predictions)

    def fit(self, X: list, y=None):
        """Fit the models based on the model type
        Returns: NeuralNetwork
        """
        if self.model_type == 'cnn':
            self.vectorizer.adapt(X)

            self.model = self.cnn(self.max_features, self.embedding_dim)
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", keras.metrics.AUC],)
        self.model.fit(X, y, epochs=self.epochs)
        return self

    def transform(self, X: list):
        """TODO: change this method"""
        return self.model.predict(X)

