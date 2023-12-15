"""
This file is used to process the data.
This includes the steps
    1. Tokenization: How?????
    2. Vectorization: How???

    Additionally: inbetween -> Move lemmatization here since we can use tokenized sentences.
"""
import os
# Python imports
import re
import pandas as pd

# Local imports
from util import log, save_file_to_path
from methods.functions import tokenize


class Tokenizer:
    TOKENIZER_ENGINES = {'regex', 'nltk', 'spacy'}

    def __init__(self, path: str, engine: str = 'spacy', save_csv: bool = True):
        """
        Given a path to a CSV file, tokenizes each row using a dataframe
        :param path: path to the CSV file
        :param engine: The selected tokenizer engine to be used. Options are 'regex', 'nltk', 'spacy'.
        """
        # Perform checks before reading the csv to reduce memory usage on false inputs :)
        if engine not in self.TOKENIZER_ENGINES:
            raise ValueError(f"Invalid tokenizer engine. Options are: {', '.join(self.TOKENIZER_ENGINES)}")

        self.path = path
        self.engine = engine
        self.df = self.read_data()

        self.spacy_nlp = None

        self.save_csv = save_csv

    def read_data(self):
        log('Reading data...')
        return pd.read_csv(self.path, engine='pyarrow')

    def run(self):
        if self.engine == 'regex':
            self.tokenize_regex()

        elif self.engine == 'nltk':
            self.tokenize_nlkt()

        elif self.engine == 'spacy':
            self.tokenize_spacy()

        else:
            raise ValueError(f"Unsupported tokenizer engine: {self.engine}")

        if self.save_csv:
            self.save()

    def tokenize_regex(self):
        log('Tokenizing the dataframe using regex...')

        def regex_find(row):
            return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", row)

        self.df['tokenized'] = self.df['post'].apply(regex_find)

    def tokenize_nlkt(self):
        log('Tokenizing the dataframe using nltk...')

        return NotImplemented

    def tokenize_spacy(self):
        log('Tokenizing the dataframe using spacy...')

        self.df['tokenized'] = tokenize.tokenize_spacy(self.df)

    def save(self):
        log('Saving tokenized dataframe...')
        self.df.to_csv(save_file_to_path(self.path, 'tokenized_extrovert.csv'), index=False)


"""
We're planning to implement two ways of vectorizing:
    CBOW model = dataset with short sentences but high number of samples (bigger dataset)
    SG model = dataset with long sentences and low number of samples (smaller dataset)
    
    from https://stackoverflow.com/questions/39224236/word2vec-cbow-skip-gram-performance-wrt-training-dataset-size
"""

class Vectorizer:
    pass


if __name__ == '__main__':
    token = Tokenizer('../data/cleaned_extrovert.csv',
                      engine='spacy',
                      save_csv=True)
    token.run()