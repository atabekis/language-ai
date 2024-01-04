"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""

from methods.process import Experiment
from methods.evaluate import Evaluate


def main():
    """Main function to run the experiment.
    Notes:
        In order to control the data-reader part of the experiment please refer to process.py;
            In the Experiment class, the Reader object can be called with the attribute "clean=True/False"
    """
    experiment = Experiment(
        time_experiments=True,
        verbose=True)
    # Comment models in order to exclude from the experiment...
    experiment.models = [
        'naive-bayes',
        'svm',
        'logistic',
        # 'random-forest',  # This lovely method takes hours -> goes in the backlog
        # 'word2vec',
        # Neural
        'cnn',
        'lstm'
    ]
    # experiment.perform_many_experiments(save_pipes=True)
    experiment.perform_single_experiment(pipeline_model='word2vec', save_pipe=True)


if __name__ == '__main__':
    # evaluator = Evaluate(pipeline_model='logistic')
    # evaluator.coefficient_weights()
    main()
