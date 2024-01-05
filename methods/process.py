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

# Resampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Classifiers
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Neural imports
from methods.neural import NeuralNetwork

# For the experiment
from methods.reader import Reader

# Cross Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

# To save the experiments
from joblib import dump, load

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


def build_pipeline(model: str, resampling_method: str = 'random-under', verbose: bool = True) -> Pipeline:
    """Used to construct a sklearn Pipeline,"""
    log(f'[Experiment] The pipeline: "{model}" selected with the resampling method: "{resampling_method}"')

    models = {
        # Naive Bayes Model wih Bag of Words
        'naive-bayes': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 2), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', MultinomialNB())
        ],
        # Support Vector Machines with tf*idf
        'svm': [
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', LinearSVC(dual='auto'))
        ],
        # Logistic Regression with tf*idf
        'logistic': [
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', LogisticRegression(random_state=__RANDOM_SEED__))
        ],
        # Neural Network models from neural.py
        # Convolutional Neural Network
        'cnn': [
            ('neural', NeuralNetwork(model_type='cnn',
                                     epochs=10,
                                     early_stop=True,
                                     verbose=verbose))
        ],
        # Long-Short Term Memory Model
        'lstm': [
            ('neural', NeuralNetwork(model_type='lstm',
                                     epochs=2,  # For me each epoch takes ~one hour, on a PC with CUDA, increase this.
                                     early_stop=True,
                                     verbose=verbose))
        ],
        # Gated Recurrent Unit Network Model
        'gru': [
            ('neural', NeuralNetwork(model_type='gru',
                                     epochs=5,
                                     early_stop=True,
                                     verbose=verbose))
        ]
    }

    selected_model = models.get(model)
    if selected_model:
        return Pipeline(selected_model, verbose=verbose)
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

    def __init__(self, time_experiments: bool = True, verbose: bool = True, debug=False):
        reader = Reader(__DATA_PATH__,
                        clean=True,
                        split=True,
                        show_info=False,
                        )

        self.time_experiments = time_experiments
        self.verbose = verbose

        self.posts = reader.posts
        self.labels = reader.labels

        self.X_train, self.y_train = reader.train[0], reader.train[1]
        self.X_test, self.y_test = reader.test[0], reader.test[1]

        self.resampling_method = 'random-under'
        self.resampling_models = ['random-over', 'random-under', 'smote', 'adasyn', 'tomek']
        """We conducted the experiments and figured that the random-under method would yield the best results"""

        self.models = ['naive-bayes', 'svm', 'logistic', 'cnn', 'lstm', 'gru']
        # ^
        """ This is passed onto perform_many_experiments, please add/remove from this list in order to conduct a 
        different experiment"""

        self.model_metrics = []
        self.model_metrics_cv = []

    def _metrics(self, y_pred: list, y_prob: list = None, cv: bool = False,
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
            'accuracy': accuracy_score(self.y_test if not cv else self.labels, y_pred),
            'precision': precision_score(self.y_test if not cv else self.labels, y_pred),
            'recall': recall_score(self.y_test if not cv else self.labels, y_pred),
            'f1_score': f1_score(self.y_test if not cv else self.labels, y_pred)}

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
            metrics_dict['roc_auc'] = roc_auc_score(self.y_test if not cv else self.labels, y_pred)

        # Round of the numbers to their 2nd decimal place in an elegant way :)
        metrics_dict = {key: format(value, '.2f') if isinstance(value, float)
                        else value for key, value in metrics_dict.items()}

        return metrics_dict

    def perform_single_experiment(self, pipeline_model: str,
                                  return_pipe: bool = False,
                                  save_pipe: bool = False,
                                  load_pipe: bool = True) -> Union[dict[str, float], Pipeline]:
        """Performs a single experiment based on the given pipeline and resampling method
        :param pipeline_model: str;
            To be passed onto the build_pipeline() function, which returns a Pipe.
        :param return_pipe: bool, default False.
            If true, only returns the pipeline -> this is used in the evaluation.
        :param save_pipe: bool, default False.
            Saves the pipeline using joblib pickle.
        :param load_pipe:
            Loads the saved pipe if it exists.
        :return Calculated metrics or the pipeline if return_pipe=True
        """
        # Check if the pipeline is already saved on local, if not perform the experiment
        dont_save_if_loaded = None
        pipeline_path = f'methods/pipelines/{pipeline_model}_{self.resampling_method}_pipeline.joblib'

        if os.path.exists(pipeline_path) and load_pipe:
            log(f'[Experiment] Existing pipeline found, loading "{pipeline_model}_{self.resampling_method}"')
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
                log(f'[Experiment] The experiment "{pipeline_model}" took {end_time - start_time:.2f} seconds.')

            if save_pipe:
                if pipeline_model in self.neural:
                    pass  # The deadline is approaching and I simply don't have time to write saving for neural...
                else:
                    log(f'[Experiment] Saving the pipeline "{pipeline_model}_{self.resampling_method}"')
                    dump(pipeline, pipeline_path)
                    log(f'[Experiment] Successfully saved the pipeline "{pipeline_model}_{self.resampling_method}"')

        if return_pipe:
            return pipeline

        y_pred, y_prob = pipeline.predict(self.X_test), None

        metrics = self._metrics(y_pred, y_prob, plot=True, pipeline_model=pipeline_model)
        self.model_metrics.append(metrics)  # Add to the class list of metrics
        if self.verbose:  # also, from init
            print(f'[{pipeline_model}] {metrics}')
        return metrics

    def perform_many_experiments(self, save_pipes: bool = False, load_pipes: bool = True) -> None:
        """Calls the perform_single_experiment with all models in self.models. Appends to the final list of metrics"""
        for model in self.models:
            self.perform_single_experiment(pipeline_model=model, save_pipe=save_pipes, load_pipe=load_pipes)
        self._export()  # Save the data for the paper

    def cross_validate_experiments(self, n_folds: int = 5, shuffle: bool = True,
                                   n_jobs: int = -1, verbose: bool = True) -> None:
        """Call each pipeline and cross validate to extract the cross validation metrics
        :param n_folds: int, optional, default 5.
            Number of cross-validation folds.
        :param shuffle: bool, optional, default True.
            Shuffle the data before splitting. The random seed is passed for reproducible results
        :param n_jobs: int, optional, default -1.
            Number of jobs to run in parallel
        :param verbose: bool, optional, default True.
            Prints out the metrics of the cross validated experiments
        """
        for model in self.models:
            try:
                neural = True if model in self.neural else False  # We cannot have multiprocessing with neural networks

                log(f'[Experiment] Cross-Validating "{model}" with {n_folds} folds...')
                pipeline = build_pipeline(model=model, resampling_method=self.resampling_method, verbose=False)

                k_fold = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=__RANDOM_SEED__)
                y_pred, y_prob = cross_val_predict(pipeline, self.posts, self.labels,
                                                   cv=k_fold, verbose=True, n_jobs=1 if neural else n_jobs), None
                metrics = self._metrics(y_pred, y_prob, cv=True, plot=False, pipeline_model=f'{model}_CV')

                if verbose:
                    print(f'[{model}] {metrics}')
                self.model_metrics_cv.append(metrics)
            except Exception as e:
                print(f'Error in model "{model}": {e}')
                self._export(cv=True, abort=True)
                return None
        self._export(cv=True)

    def _export(self, cv=False, abort: bool = False):
        """Takes the finalized model metrics and exports it as .tex table"""
        import pandas as pd
        dataframe = pd.DataFrame(self.model_metrics)
        if not cv:
            dataframe.to_latex(f'{__EXPERIMENTS_PATH__}/many_experiments.tex', index=False)
            log('[Experiment] Successfully saved "many_experiments.tex"')
        elif cv and abort:
            dataframe.to_latex(f'{__EXPERIMENTS_PATH__}/many_experiments_CV_abort.tex', index=False)
            log('[Experiment] Successfully saved "many_experiments_CV.tex"')
        else:
            dataframe.to_latex(f'{__EXPERIMENTS_PATH__}/many_experiments_CV.tex', index=False)
            log('[Experiment] Successfully saved "many_experiments_CV.tex"')


if __name__ == '__main__':
    experiment = Experiment(
        time_experiments=True,
        verbose=True)
    experiment.perform_many_experiments()
