# Methods
>In this file we list the various methods found under the `/methods` directory.
## Reader 
`reader.py` \
This file supports the cleaning and representation of the dataset.
### Objects
The `Dataset` class:
* This class is used to load and represent the _uncleaned_ dataset.
* Methods:
  * Changing the column names
    * `auhtor_ID` → `author_id`
    * `extrovert` → `label`
  * Check class imbalance (imbalance on the targets)
  * Author counts

The `CleanData` class:
* Inherits from the Dataset class and uses various class methods to clean the dataset.
* Methods:
  * Remove uppercase letters.
  * Remove (some) punctuation.
  * Remove stopwords.
  * Decode HTML attributes.
* Saves the cleaned file under `data/cleaned_extrovert.csv`

The `Reader` class:
* Reads the clean data and splits it into test/train using [scikit-learn](https://scikit-learn.org/)'s train_test_split()

## Neural
`neural.py`
### Objects
The `NeuralNetwork` class:
* This class is used to make predictions using neural network algorithms by [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

The class parameters/hyperparameters can be changed within the `__init__` method:
```python
def __init__(self,
             model_type: str = 'cnn',
             max_features: int = 20000,
             embedding_dim: int = 128,
             sequence_length: int = 500,
             epochs: int = 3,
             early_stop: bool = False) -> None:
```
By default, the selected model will be a Convolutional Neural Network (CNN)
#### Convolutional Neural Network
The layers of the CNN are set as follows:
```python
 def cnn(self, max_features: int, embedding_dim: int) -> keras.Model:
        """Representation of a Convolutional Neural Network."""
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
```
>The inspiration for network can be found [here](https://keras.io/examples/nlp/text_classification_from_scratch/)
#### Long Short-Term Memory Network
One other network found in the `NeuralNetwork` class is the Long Short-Term Memory network or (LSTM).
Hence, the layers of the LSTM:
```python
    def lstm(self, max_features: int, embedding_dim: int, lstm_dropout: float = 0.2) -> keras.Model:
        """
        Representing a Long Short-Term Memory Network"""

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
```
>The inspiration and huge thanks goes to [here](https://medium.com/mlearning-ai/the-classification-of-text-messages-using-lstm-bi-lstm-and-gru-f79b207f90ad) and [here](https://www.analyticsvidhya.com/blog/2021/06/lstm-for-text-classification/).
#### Early Stopping
Setting `early_stop=True` will result in the network automatically stopping if there's no improvement over the loss.

