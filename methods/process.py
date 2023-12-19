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
from functools import lru_cache, cache

# NLP imports
from sklearn.feature_extraction.text import CountVectorizer


# Local imports
from util import log, save_file_to_path
from methods.functions import tokenize


class Tokenizer:
    TOKENIZER_ENGINES = {'regex', 'nltk', 'spacy'}

    def __init__(self, path: str, engine: str = 'spacy', save_csv: bool = True, drop_original: bool = True):
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

        self.drop_original = drop_original
        self.save_csv = save_csv

    @cache
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

        if self.drop_original:
            self.drop_original_column()

        if self.save_csv:
            self.save()

    def tokenize_regex(self):
        log('Tokenizing the dataframe using regex...')

        def regex_find(row):
            return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", row)

        self.df['tokens'] = self.df['post'].apply(regex_find)

    def tokenize_nlkt(self):
        log('Tokenizing the dataframe using nltk...')

        return NotImplemented

    def tokenize_spacy(self):
        log('Tokenizing the dataframe using spacy...')

        self.df['tokens'] = tokenize.tokenize_spacy(self.df)

    def drop_original_column(self):
        log('Dropping the original column...')
        self.df = self.df.drop('post', axis=1)

    def save(self):
        log('Saving tokenized dataframe...')
        self.df.to_csv(save_file_to_path(self.path, 'tokenized_extrovert.csv'), index=False)


"""
Found the holy bible of vectorizing:
https://neptune.ai/blog/vectorization-techniques-in-nlp-guide
"""


class Vectorizer:

    VECTORIZER_ENGINES = {'bow', 'tf-idf', 'word2vec'}

    def __init__(self, path, engine='bow'):
        if engine not in self.VECTORIZER_ENGINES:
            raise ValueError(f"Invalid vectorizer engine. Options are: {', '.join(self.VECTORIZER_ENGINES)}")

        self.path = path
        self.engine = engine
        self.df = self.read_data()

    def read_data(self):
        log('Reading data...')
        return pd.read_csv(self.path, engine='pyarrow')

    def vectorize(self):
        return NotImplemented

    def bow_vectorize(self):
        log('Vectorizing using: Bag of Words...')


        vectorizer = CountVectorizer(stop_words='english')
        x = vectorizer.fit_transform(
            self.df['post'].apply(lambda x: ' '.join(x)))
        feature_names = vectorizer.get_feature_names_out()

        dtm_df = pd.DataFrame(x.toarray(), columns=feature_names)
        return dtm_df



if __name__ == '__main__':
    vectorizer = Vectorizer('../data/tokenized_extrovert.csv')
    print(vectorizer.bow_vectorize())