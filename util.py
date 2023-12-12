# Core imports
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count

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


def parallel_lemmatize(data, batch_size=1):
    with Pool(processes=cpu_count() - 1) as pool:
        results = list(
            tqdm(pool.imap(lemmatize_text, [data[i:i + batch_size] for i in range(0, len(data), batch_size)]),
                 total=len(data)))
    return results


# Testing framework here:
if __name__ == '__main__':
    pass
