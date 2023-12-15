# Core imports
import os
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from threading import Thread

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


def lemmatize_text(texts):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    lemmatized_texts = []
    for text in texts:
        doc = nlp(text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        lemmatized_texts.append(lemmatized_text)
    return lemmatized_texts


def parallel_lemmatize(data):
    """

    :param data:
    :return:
    """
    processors = os.cpu_count()
    batches = np.array_split(data, processors)

    with ProcessPoolExecutor(processors) as executer:
        result = pd.concat(executer.map(lemmatize_text, batches))
    return result
    # with Pool(processes=cpu_count()l - 1) as pool:
    #     results = list(
    #         tqdm(pool.imap(lemmatize_text, [data[i:i + batch_size] for i in range(0, len(data), batch_size)]),
    #              total=len(data)))
    # return results


def split_dataframe(df, n):
    """
    Split a DataFrame or Series into 'n' pieces.

    :param df: pd.DataFrame or pd.Series; The input DataFrame or Series.
    :param n: int Number of pieces to split the DataFrame or Series into.
    :return: List of pd.DataFrame or pd.Series; List containing 'n' pieces of the input DataFrame or Series.
    """
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise ValueError("Input must be a DataFrame or Series.")

    total_rows = len(df)
    rows_per_chunk = total_rows // n
    remainder = total_rows % n

    chunks = []
    start = 0

    for i in range(n):
        chunk_size = rows_per_chunk + (1 if i < remainder else 0)
        end = start + chunk_size
        chunks.append(df.iloc[start:end])
        start = end

    return chunks


def reconstruct_dataframe(chunks):
    """
    Reconstruct the original DataFrame or Series from the split pieces.

    :param chunks: List of pd.DataFrame or pd.Series; List containing pieces of the original DataFrame or Series
    return: pd.DataFrame or pd.Series; The reconstructed DataFrame or Series.
    """
    if not all(isinstance(chunk, (pd.DataFrame, pd.Series)) for chunk in chunks):
        raise ValueError("All elements in the input list must be DataFrames or Series.")

    return pd.concat(chunks, ignore_index=True)


# Testing framework here:
if __name__ == '__main__':
    pass
