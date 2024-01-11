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


if __name__ == '__main__':
    main()
