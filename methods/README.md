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
    x = LSTM(128, return_sequences=True)(x)  # No dropout here
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    out_layer = Dense(1, activation="sigmoid", name="predictions")(x)
    return Model(text_input, out_layer)
```
>The inspiration and huge thanks goes to [here](https://medium.com/mlearning-ai/the-classification-of-text-messages-using-lstm-bi-lstm-and-gru-f79b207f90ad) and [here](https://www.analyticsvidhya.com/blog/2021/06/lstm-for-text-classification/).
#### Early Stopping
Setting `early_stop=True` will result in the network automatically stopping if there's no improvement over the loss.

## Process
`process.py`
This is the final .py file of the project where everything comes together to conduct the experiments.
### Objects
The `Experiment` class:
* This class conducts the experiments using two functions `perform_single_experiment()` and `perform_many_experiments()`.
* The function `build_pipeline` is called to create a [sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html):
```python
# Example pipeline model in the build_pipeline function
'naive-bayes': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 3), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', MultinomialNB())]
```
* The function `resampler` returns a resampling method implemented from the [imbalanced-learn](https://imbalanced-learn.org/stable/index.html) library.
```python
 models = {
        'random-over': RandomOverSampler(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'random-under': RandomUnderSampler(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'smote': SMOTE(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'adasyn': ADASYN(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'tomek': TomekLinks(sampling_strategy='auto')
    }
```
* The pipeline is fitted withing the class methods and then the metrics are extracted from the predicted and real values on the split data.
* The metrics of each classifier & network are saved as a .tex file under `output`
* TODO: add table → findings

## Tune-Models
`tune_models.py`
This file is to discover the hyperparameter tuning for sklearn classifiers.

### Objects
The `Tuner` class:
* This class includes methods for (some) of the classifiers as functions.
* The fitting is carried through [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) implemented from sklearn.
* The `_grid_seach_tuner()` method; builds the pipeline from `process.py`, carries the k-fold-CV and extracts the best score and best parameters.
```python
def _grid_search_tuner(self, model_name: str, param_grid: dict[any], cv: int = 5):
    """Builds the pipeline and performs 5-fold cross validation to find the best model parameters
    :param model_name: str
        name of the model from: naive-bayes, svm, random-forest, logistic
    :param param_grid:
        pre-fitted parameters for the model
    :param cv:
        k value for the k-fold cross validation
    """
    pipeline = build_pipeline(model=model_name, resampling_method=self.resampling_method, verbose=True)

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=self.scoring,
                               n_jobs=2, verbose=True)
    grid_search.fit(self.X_train, self.y_train)

    print(f'Best score: {grid_search.best_score_:%0.3f}')
    print('Best parameters set:')
    best_params = grid_search.best_params_
    for param in sorted(param_grid.keys()):
        print("\t%s: %r" % (param, best_params[param]))
    return best_params
```
* Then this method is used in `tune_X` where `X` are the classifiers. The methods are initialized with dummy hyperparameters.
```python
def tune_rf(self):
    model_name = 'random-forest'
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]}

    return self._grid_search_tuner(model_name=model_name, param_grid=param_grid)

def tune_svm(self):
    model_name = 'svm'
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__loss': ['hinge', 'squared_hinge'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__dual': [True, False]
    }
    return self._grid_search_tuner(model_name=model_name, param_grid=param_grid)
```