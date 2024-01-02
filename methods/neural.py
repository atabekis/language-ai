"""Neural network models: CNN"""
# Python imports & setting the backend
import os;

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3';
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

# Neural imports
import keras
import tensorflow as tf  # This is just for tf.string on line 58, can be changed if there's any other option
from keras import Model, Input
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Reshape

# We need to inherit BaseEstimator and TransformerMixin in order to use the class in a Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Util
from util import log


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
    :param early_stop: bool, optional, default: False
        Stop the training if the model has stopped improving.
    Notes:
    -----
    Some (a lot of) inspiration and structuring from
    https://github.com/cmry/amica/blob/master/neural.py
    """

    def __init__(self,
                 model_type: str = 'cnn',
                 max_features: int = 20000,
                 embedding_dim: int = 128,
                 sequence_length: int = 500,
                 epochs: int = 3,
                 batch_size: int = 50,
                 early_stop: bool = False) -> None:
        """Initialize the basic control parameters of the neural network"""

        self.model_type = model_type

        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop  # Bool statement to control self.callback -> passed onto model.fit

        self.model = None
        self.history = None  # It's nice to keep this if we want to plot loss/AUC/accuracy over epochs
        self.callback = None  # ^TODO: add:: when debug -> early_stop=True

        self.vectorizer = TextVectorization(  # This is to make sure our vectors are padded and properly vectorized
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length)

        self.predictions = None  # Testing - to remove later


    def cnn(self, max_features: int, embedding_dim: int) -> keras.Model:
        """Representation of a Convolutional Neural Network.

        Notes:
            Implementation from:
            https://keras.io/examples/nlp/text_classification_from_scratch/
        """
        log('Fitting the Neural Network: Convolutional Neural Network')
        text_input = Input(shape=(1,), dtype=tf.string, name='text')
        x = self.vectorizer(text_input)  # We vectorize the text here

        # Start network here
        x = Embedding(max_features + 1, embedding_dim)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        out_layer = Dense(1, activation="sigmoid", name="predictions")(x)  # Sigmoid for binary output
        return Model(text_input, out_layer)

    def lstm(self, max_features: int, embedding_dim: int, lstm_dropout: float = 0.2) -> keras.Model:
        """
        Representing a Long Short-Term Memory Network
        Notes:
            Implementation from these two lovely guides:
            https://medium.com/mlearning-ai/the-classification-of-text-messages-using-lstm-bi-lstm-and-gru-f79b207f90ad
            https://www.analyticsvidhya.com/blog/2021/06/lstm-for-text-classification/
        """
        log('Fitting the Neural Network: Long Short-Term Memory Network')

        text_input = Input(shape=(1,), dtype=tf.string, name='text')
        x = self.vectorizer(text_input)

        # Start network structure here ::[L128, L128, D128, D1]
        x = Embedding(max_features, embedding_dim)(x)
        x = Reshape((-1, embedding_dim))(x)
        x = LSTM(128, return_sequences=True, dropout=lstm_dropout)(x)
        x = LSTM(128, return_sequences=True)(x)  # No dropout here?
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)

        out_layer = Dense(1, activation="sigmoid", name="predictions")(x)
        return Model(text_input, out_layer)

    def fit(self, X: list, y=None):  # -> NeuralNetwork
        """Fit the models based on the model type
        Returns: NeuralNetwork
        """
        models = {
            'cnn': self.cnn,
            'lstm': self.lstm
        }

        fit_model = models.get(self.model_type)  # Using .get does not throw an error when wrong model (english????)
        if fit_model:
            self.vectorizer.adapt(X)  # Vectorizer needs to be initialized
            self.model = fit_model(self.max_features, self.embedding_dim)

        if self.early_stop:  # Early stop control
            split_index = int(len(X) * 0.8)  # The following two lines are used to split the current X into validation
            X_val, y_val = np.array(X[:split_index]), np.array(y[:split_index])
            validation_data = np.array(X_val[-split_index:]), np.array(y_val[-split_index:])
            self.callback = [EarlyStopping(monitor='val_loss', patience=1)]

        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", keras.metrics.AUC], )
        self.history = self.model.fit(X, y,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      validation_data=validation_data if self.early_stop else None,
                                      callbacks=[self.callback] if self.early_stop else None)

        return self

    # def predict(self, X: list):
    #     """Based on sklearn API"""
    #     return self.model.predict(X)  # Store later for predict proba

    def predict(self, X: list) -> list:
        """Predict using the trained model
        Notes:
            Implementation also from
            https://github.com/cmry/amica/blob/master/neural.py#L228
        """
        # y_hat = np.argmax(self.model.predict(X), axis=1)
        # return [int(y_i) for y_i in y_hat]  # Mr. Emmery's implementation
        # prob = np.reshape(pred, (-1, 2))
        # return prob
        y_hat = self.model.predict(X)
        return (y_hat > 0.5).astype(int)  # My implementation


