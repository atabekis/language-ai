"""
In this .py file we lay out the pipelines for the various methods of
    1. Tokenization
    2. Vectorization
    3. Regression/model
"""
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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

    }
    selected_model = models.get(pipeline_model)
    if selected_model:
        return Pipeline([
            ('vectorizer', selected_model['vectorizer']),
            ('classifier', selected_model['classifier'])
        ])
    else:
        raise KeyError(f"Invalid key: {pipeline_model}. Choose from {list(models.keys())}.")


if __name__ == '__main__':
    pipeline = build_pipeline('logisic')
