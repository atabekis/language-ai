import os
__DATA_PATH__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'extrovert_introvert.csv')
__EXPERIMENTS_PATH__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'methods', 'output')
__PIPELINES_PATH__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'methods', 'pipelines')
if os.getcwd() == 'project':  # If not main.py
    # __PROJECT_PATH__ = os.path.dirname(os.path.realpath(__file__))
    __PROJECT_PATH__ = os.path.dirname(os.getcwd())

else:
    __PROJECT_PATH__ = os.path.dirname(os.path.realpath(__file__))
