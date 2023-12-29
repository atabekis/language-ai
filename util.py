# Core imports
import os
from datetime import datetime

# Metrics


def log(*args, **kwargs):
    """
    I'm using this to print whatever I want with [timestamp] at the beginning of each print
    :return: Timestamped print message
    """
    print(f'[{datetime.now().strftime("%H:%M:%S:%f")[:-3]}]', *args, **kwargs)


def save_file_to_path(path: str, filename: str) -> os.path:
    """
    This is a half deprecated function. Used in the wrapper classes to save files, whether they're run through main.py
    or individually.
    :param path: path to the file to save
    :param filename: name of the file to save
    :return: path of the saved file
    """
    directory, _filename = os.path.split(path)
    return os.path.join(directory, filename)



# Testing framework here:
if __name__ == '__main__':
    pass
