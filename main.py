"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""

from methods.process import Experiment


def main(
        # Main experiments
        single_experiment: str = None,
        multiple_experiments: bool = True,
        cross_validate_experiments: bool = False,
        # Controls for the experiment class
        time_experiments: bool = True,
        verbose: bool = True,
        debug: bool = False,
        # Saving and loading models
        load_existing_models: bool = False,
        save_models: bool = True):
    """Main function to run the experiment.
    Notes:
        In order to control the data-reader part of the experiment please refer to process.py;
            In the Experiment class, the Reader object can be called with the attribute "clean=True/False"

    :param single_experiment: str, default None,
        Pss one of the six experiments to perform a single experiment.
    :param multiple_experiments: bool, default True
        Runs all the experiments
    :param cross_validate_experiments: bool, default False,
        Runs the simple models with k-fold cross validation.
    :param time_experiments: bool, default True,
        Prints the time it took to execute the experiment.
    :param verbose: bool, default True,
        Verbose output.
    :param debug: bool, default False,
        Cuts the test and train data by a debug cutoff factor (0.1) - used to test new models.
    :param load_existing_models: bool, default False,
        loads existing pipelines from the /pipelines folder.
    :param save_models: bool, default True,
        saves the fitted pipelines.
    """
    # Create the experiment object
    experiment = Experiment(
        time_experiments=time_experiments,
        verbose=verbose,
        debug=debug)

    if single_experiment:
        experiment.perform_single_experiment(pipeline_model=single_experiment,
                                             save_pipe=save_models,
                                             load_pipe=load_existing_models)
    if multiple_experiments:
        experiment.perform_many_experiments(save_pipes=save_models, load_pipes=load_existing_models)

    if cross_validate_experiments:
        experiment.cross_validate_experiments(verbose=verbose)


# Main
if __name__ == '__main__':
    main(
        multiple_experiments=True,
        time_experiments=True,
        verbose=True,
    )
