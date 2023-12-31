"""Main file for the experiment/process through pipelines"""
# Python imports
import time
from typing import Union

# Importing the pipeline
# from sklearn.pipeline import Pipeline --> this was breaking with resampling, but i'll keep it here
from imblearn.pipeline import Pipeline

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Resampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Neural imports
from methods.neural import NeuralNetwork

# For the experiment
from methods.reader import Reader

# Util
from util import log
from config import __DATA_PATH__, __EXPERIMENTS_PATH__


__RANDOM_SEED__ = 5


def resampler(model: str = 'random-under') -> Union[RandomOverSampler, RandomUnderSampler,
SMOTE, ADASYN, TomekLinks, None]:
    """
    Function to select a resampling method based on models from the imbalanced-learn library.

    :param model: str, default 'random-under'
    :return:
    """
    models = {
        'random-over': RandomOverSampler(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'random-under': RandomUnderSampler(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'smote': SMOTE(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'adasyn': ADASYN(sampling_strategy='auto', random_state=__RANDOM_SEED__),
        'tomek': TomekLinks(sampling_strategy='auto')
    }
    if model:
        return models.get(model)
    else:
        return None


def build_pipeline(model: str, resampling_method: str = 'random-under') -> Pipeline:
    """Used to construct a sklearn Pipeline,"""
    print('\n')  # To separate the cleaning output from the model outputs
    log(f'The pipeline: "{model}" selected with the resampling method: "{resampling_method}"')

    models = {
        # Naive Bayes Model wih Bag of Words
        'naive-bayes': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 3), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', MultinomialNB())
        ],
        # Support Vector Machines with tf*idf
        'svm': [
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', SVC())
        ],
        # Logistic Regression with tf*idf
        'logistic': [
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', LogisticRegression(random_state=__RANDOM_SEED__))
        ],
        # Random forest model with bag of words
        'random-forest': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 3), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', RandomForestClassifier(random_state=__RANDOM_SEED__))
        ],

        # Neural Network models from neural.py
        # Convolutional Neural Network
        'cnn': [
            ('neural', NeuralNetwork(model_type='cnn'))
        ],
        # Long-Short Term Memory Model
        'lstm': [
            ('neural', NeuralNetwork(model_type='lstm'))
        ]
    }

    selected_model = models.get(model)
    if selected_model:
        return Pipeline(selected_model)
    else:
        raise KeyError(f'Invalid key: {model}. Please choose from {list(models.keys())}')


class Experiment:
    """
    Main and (hopefully) last class, this combines all the other object and performs experiment(s). The metrics are
    computed and exported to be used in the paper.

    :param time_experiments: bool, default=True
        times each experiment and prints the time it took to complete the task.
    :param verbose: bool, default=True
        if True, prints the status of the experiment
    """

    def __init__(self, time_experiments: bool = True, verbose: bool = True):
        reader = Reader(__DATA_PATH__,
                        clean=True,
                        split=True,
                        show_info=False)

        self.time_experiments = time_experiments
        self.verbose = verbose

        self.posts = reader.posts
        self.labels = reader.labels

        self.X_train, self.y_train = reader.train[0], reader.train[1]
        self.X_test, self.y_test = reader.test[0], reader.test[1]

        self.resampling_method = 'random-under'
        """We conducted the experiments and figured that the random-under method would yield the best results"""

        self.models = ['naive-bayes', 'svm', 'logistic', 'random-forest', 'cnn', 'lstm']
        # ^
        """ This is passed onto perform_many_experiments, please add/remove from this list in order to conduct a 
        different experiment"""

        self.model_metrics = []

    def _metrics(self, y_pred: list, y_prob: list = None,
                 plot: bool = True, pipeline_model: str = None) -> dict[str, float]:
        """
        Function to compute various binary classification metrics.

        :param y_pred: array-like
            predicted labels from the pipeline
        :param y_prob: array-like, optional (default = None)
            predicted probabilities
        :param plot: bool, optional (default = True)
            whether to plot the ROC curve
        :param pipeline_model: str, optional (default = None)
            used for the title of the roc plot

        :return:
            dictionary containing computed metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        import matplotlib.pyplot as plt

        metrics_dict = {
            'model': pipeline_model,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred), 'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred)}

        # ROC-AUC
        if y_prob is not None:
            metrics_dict['roc_auc'] = roc_auc_score(self.y_test, y_prob)

            # ROC Curve
            if plot:
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for "{pipeline_model}"')
                plt.legend(loc='lower right')
                plt.show()

        # Round of the numbers to their 2nd decimal place in an elegant way :)
        metrics_dict = {metric: format(value, '.2f') for metric, value in metrics_dict.items() if isinstance(value, float)}

        return metrics_dict

    def perform_single_experiment(self, pipeline_model: str) -> dict[str, float]:
        """Performs a single experiment based on the given pipeline and resampling method
        :param pipeline_model: str,
            to be passed onto the build_pipeline() function, which returns a Pipe
        """
        # Start timing
        start_time = time.time()
        # We build the pipeline
        pipeline = build_pipeline(pipeline_model, resampling_method=self.resampling_method)

        pipeline.fit(self.X_train, self.y_train)  # Fit the model
        y_pred, y_prob = pipeline.predict(self.X_test), pipeline.predict_proba(self.X_test)[:, 1]
        # End timing
        end_time = time.time()

        if self.time_experiments:  # from init
            log(f'Experiment "{pipeline_model}" took {end_time-start_time:.2f} seconds.')

        metrics = self._metrics(y_pred, y_prob, plot=True, pipeline_model=pipeline_model)
        self.model_metrics.append(metrics)  # Add to the class list of metrics
        if self.verbose:  # again, from init
            print(metrics)
        return metrics

    def perform_many_experiments(self) -> None:
        """Calls the perform_single_experiment with all models in self.models. Appends to the final list of metrics"""
        for model in self.models:
            self.perform_single_experiment(pipeline_model=model)
        self._export()  # Save the data for the paper

    def _export(self):
        """Takes the finalized model metrics and exports it as .tex table"""
        import pandas as pd
        dataframe = pd.DataFrame(self.model_metrics)
        dataframe.to_latex(f'{__EXPERIMENTS_PATH__}/many_experiments.tex', index=False)
        log('Successfully saved "many_experiments.tex"')


if __name__ == '__main__':
    experiment = Experiment(
        time_experiments=True,
        verbose=True)
    experiment.perform_many_experiments()
