"""Main file for the experiment/process through pipelines"""
# Python imports
import os
import time
from typing import Union

# Importing the pipeline
# from sklearn.pipeline import Pipeline --> this was breaking with resampling, but i'll keep it here jic
from imblearn.pipeline import Pipeline

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from util import TfidfVectorizerTQDM  # Custom vectorizer with a nice progress bar...

# Classifiers
from sklearn.svm import SVC, LinearSVC
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

# To save the experiments
from joblib import dump, load

# Util
from util import log
from config import __DATA_PATH__, __EXPERIMENTS_PATH__, __PIPELINES_PATH__

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


def build_pipeline(model: str, resampling_method: str = 'random-under', verbose: bool = True) -> Pipeline:
    """Used to construct a sklearn Pipeline,"""
    # print('\n')  # To separate the cleaning output from the model outputs
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
            ('vectorizer', TfidfVectorizerTQDM(ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            # ('classifier', SVC())
            ('classifier', LinearSVC(dual='auto', verbose=verbose))
        ],
        # Logistic Regression with tf*idf
        'logistic': [
            ('vectorizer', TfidfVectorizerTQDM(ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', LogisticRegression(random_state=__RANDOM_SEED__, verbose=verbose))
        ],
        # Random forest model with bag of words
        'random-forest': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 3), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', RandomForestClassifier(random_state=__RANDOM_SEED__, verbose=verbose))
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
    # TODO: There are a few steps i can take to optimize the process and we don't use word
    # ^embeddings. Check out: https://chat.openai.com/c/07776441-c2e6-46bd-9b84-9811ecc9c101

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
    TODO: Add cross validation
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
        else:
            metrics_dict['roc_auc'] = roc_auc_score(self.y_test, y_pred)

        # Round of the numbers to their 2nd decimal place in an elegant way :)
        metrics_dict = {metric: format(value, '.2f') for metric, value
                        in metrics_dict.items() if isinstance(value, float)}

        return metrics_dict

    def perform_single_experiment(self, pipeline_model: str,
                                  return_pipe: bool = False,
                                  save_pipe: bool = False) -> Union[dict[str, float], Pipeline]:
        """Performs a single experiment based on the given pipeline and resampling method
        :param pipeline_model: str,
            to be passed onto the build_pipeline() function, which returns a Pipe
        :param return_pipe: bool, default False,
            if true, only returns the pipeline -> this is used in the evaluation
        :param save_pipe: bool, default False
        """
        # Check if the pipeline is already saved on local, if not perform the experiment
        dont_save_if_loaded = None
        pipeline_path = f'methods/pipelines/{pipeline_model}_{self.resampling_method}_pipeline.joblib'

        if os.path.exists(pipeline_path):
            log(f'Existing pipeline found, loading "{pipeline_model}"')
            pipeline = load(pipeline_path)
            dont_save_if_loaded = True
        else:
            # Start timing
            start_time = time.time()
            # We build the pipeline
            pipeline = build_pipeline(pipeline_model, resampling_method=self.resampling_method)

            pipeline.fit(self.X_train, self.y_train)  # Fit the model

            end_time = time.time()  # End timing

            if self.time_experiments:  # from init
                log(f'Experiment "{pipeline_model}" took {end_time - start_time:.2f} seconds.')

            if save_pipe:
                if pipeline_model == 'cnn' or pipeline_model == 'lstm':
                    pass
                else:
                    log(f'Saving the pipeline "{pipeline_model}"')
                    dump(pipeline, pipeline_path)
                    log(f'Successfully saved the pipeline "{pipeline_model}"')

        if return_pipe:
            return pipeline

        y_pred, y_prob = pipeline.predict(self.X_test), None

        metrics = self._metrics(y_pred, y_prob, plot=True, pipeline_model=pipeline_model)
        self.model_metrics.append(metrics)  # Add to the class list of metrics
        if self.verbose:  # also, from init
            print(metrics)
        return metrics

    def perform_many_experiments(self, save_pipes: bool = False) -> None:
        """Calls the perform_single_experiment with all models in self.models. Appends to the final list of metrics"""
        for model in self.models:
            self.perform_single_experiment(pipeline_model=model, save_pipe=save_pipes)
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
