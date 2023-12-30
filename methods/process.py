"""Main file for the experiment/process through pipelines"""
# Importing the pipeline
from sklearn.pipeline import Pipeline

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Neural imports
from methods.neural import NeuralNetwork


# To delete later
import pandas as pd
from sklearn.model_selection import train_test_split


# For passing onto all random states:
__RANDOM_SEED__ = 5

# Helper for the build_pipeline
model_keys = ['naive-bayes', 'svm', 'logistic', 'random-forest']


def build_pipeline(pipeline_model: str) -> Pipeline:
    """
    Takes the type of pipeline as parameter and returns a sklearn Pipeline.
    :param pipeline_model: TODO: add the types to docstring
    :return: sklearn Pipeline
    """
    models = {
        # Basic models go here: nb, svm, logit, rf
        'naive-bayes': {
            'vectorizer': CountVectorizer(
                ngram_range=(1, 3),
                binary=True  # TODO: experiment with this statement
            ),
            'classifier': MultinomialNB()
        },
        'svm': {
            'vectorizer': TfidfVectorizer(
                ngram_range=(1, 3),
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,  # this is what the lecturer showed us: 1+log(tf)
            ),
            'classifier': SVC()
        },
        'logistic': {
            'vectorizer': TfidfVectorizer(
                ngram_range=(1, 3),
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
            ),
            'classifier': LogisticRegression(
                random_state=__RANDOM_SEED__
            ),
        },
        'random-forest': {
            'vectorizer': CountVectorizer(
                ngram_range=(1, 3),
                binary=True  # TODO: experiment with this statement
            ),
            'classifier': RandomForestClassifier(
                random_state=__RANDOM_SEED__
            )
        },
        'cnn': {
            'neural': NeuralNetwork(
                model_type='cnn'
            )
        },
        'lstm': {
            'neural': NeuralNetwork(
                model_type='lstm',
                # epochs=1,  # TODO: DELETE THESE!!
                # early_stop=True  # ALSO THIS
            )
        }
    }

    # Select the model using .get -> eval to True/False
    selected_model = models.get(pipeline_model)
    if selected_model:
        if 'vectorizer' in selected_model and 'classifier' in selected_model:
            return Pipeline([
                ('vectorizer', selected_model['vectorizer']),
                ('classifier', selected_model['classifier'])
            ])
        elif 'neural' in selected_model:
            return Pipeline([
                ('neural', selected_model['neural'])])

        else:
            raise KeyError(f"Invalid key: {pipeline_model}. Choose from {list(models.keys())}.")
    else:
        raise KeyError(f"Invalid key: {pipeline_model}. Choose from {list(models.keys())}.")


class Experiment:
    pass


if __name__ == '__main__':
    pipeline = build_pipeline('lstm')
    df = pd.read_csv('../data/cleaned_extrovert.csv', engine='pyarrow')
    x_train, x_test, y_train, y_test = train_test_split(df['post'], df['label'],
                                                        test_size=0.2, random_state=__RANDOM_SEED__)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

