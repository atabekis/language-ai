"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""


from methods.process import Experiment


def main():
    """Main function to run the experiment."""
    experiment = Experiment(
        time_experiments=True,
        verbose=True)
    experiment.perform_many_experiments()
    # experiment.perform_single_experiment(pipeline_model='svm')


if __name__ == '__main__':
    main()
