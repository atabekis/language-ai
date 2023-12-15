# Core imports
import os
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

# NLP imports
import spacy


def log(*args, **kwargs):
    """
    I'm using this to print whatever I want with [timestamp] at the beginning of each print
    :return: Timestamped print message
    """
    print(f'[{datetime.now().strftime("%H:%M:%S:%f")[:-3]}]', *args, **kwargs)


def save_file_to_path(path, filename):
    directory, _filename = os.path.split(path)
    return os.path.join(directory, filename)


# Testing framework here:
if __name__ == '__main__':
    pass
