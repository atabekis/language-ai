"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""

from methods.process import Experiment


def main():
    """Main function to run the experiment.
    Notes:
        In order to control the data-reader part of the experiment please refer to process.py;
            In the Experiment class, the Reader object can be called with the attribute "clean=True/False"
    """
    experiment = Experiment(
        time_experiments=True,
        verbose=True,
        debug=False  # This cuts the data by a debug factor.
    )
    experiment.debug_cutoff = 0.1
    # Comment models in order to exclude from the experiment...
    experiment.models = [
        'naive-bayes',
        'svm',
        'logistic',
        # Neural
        'cnn',
        'lstm',
        'fasttext',
        # 'gru'  # Does not work
    ]
    experiment.perform_single_experiment(pipeline_model='fasttext', save_pipe=False, load_pipe=False)
    # experiment.perform_many_experiments(save_pipes=False, load_pipes=True)
    # experiment.cross_validate_experiments()


if __name__ == '__main__':
    # evaluator = Evaluate(pipeline_model='logistic')
    # evaluator.coefficient_weights()
    main()
