# Python imports
import os
import time
import numpy as np
import pandas as pd
from typing import List

# Other
import spacy
from tqdm import tqdm
from datetime import timedelta

# Local
from util import split_dataframe

"""
Findings:

1. Spacy:
    Test Case 1: n_processes = 3, no batches, numpy array
        Memory: stable around 12 GB
        Time taken: 21 Minutes
    
    Test Case 2: n_processes = 3, batch_size = len(df) // 3, numpy array
        Crashed my computer -> ran out of memory :))
        
    Test Case 3: n_processes = 4, batch_size = 100, numpy array
        Memory: wavy around 12-18GB
        CPU: 75% usage
        Time taken: got bored, but bit faster than Test Case 1
        
    Test Case 4: n_processes = 8, batch_size = 200, numpy array
        Time taken: got 15-16 items per second
        
    Test Case 5: n_processes = 8, batch_size = 300, numpy array
        Time taken: got 15-16 items per second

        
Some Notes:
Each process require its own memory, every time a new process is created, model data has to be copied into memory
for every process. If doing small tasks -> increase batch size and lower down number of processes.   
    
My prayers have been accepted, and I found this holy document:
https://stackoverflow.com/questions/74181750/a-checklist-for-spacy-optimization?rq=3
"""

nlp = spacy.load("en_core_web_sm", disable=['tagger', 'ner', 'lemmatizer', 'textcat'])


def tokenize_spacy(df: pd.DataFrame) -> List[List[str]]:
    """
    This function tokenizes the 'post' column of the dataframe using the spacy module.
    Multiprocessing across all cores is utilized.

    Notes: CPU usually throttles as we're loading the first batch and the nlp.pipe, then usage is reduced!

    This function is also used to make your CPU warm and cozy, personally it's a win-win since I'll spend less money
    on gas as long as this function runs.
    :param df: dataframe
    :return: tokenized array of the 'post' column
    """
    try:
        texts_array = np.array(df['post'])
        store = []

        start = time.monotonic()

        for doc in tqdm(nlp.pipe(texts_array, batch_size=250, n_process=os.cpu_count()), total=len(texts_array),
                        desc='This will take a while, get yourself some coffee :)'):

            tokens = [token.text for token in doc]
            store.append(tokens)

        end = time.monotonic()
        print(f'Time took: {timedelta(seconds=end - start)}')
        return store

    except UserWarning:
        pass


if __name__ == '__main__':
    df = pd.read_csv('../../data/cleaned_extrovert.csv', engine='pyarrow')
    print(tokenize_spacy(df))


