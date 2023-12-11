"""
We have the following process:
1. Dataset
2. Understand data -> explore_data.ipynb
3. Data preparation
    a. Group data -> 2 dimensions (Extroverted: 1, Introverted: 0)
    b. Data cleaning:
        i. Lowercase
        ii. Remove link
        iii. Punctuation
        iv. Stopwords
    c. Lemmatization
4. Data processing:
    a. Tokenization -> punk? (other options)
    b. Word2Vec -> CBOW (other options?)
5. Supervised learning -> ???
6. Improvement -> ?? (Handling imbalanced data -> SMOTE)
7. Model comparison & Model evaluation
"""

# Core imports
import time
import string
from datetime import datetime
import pandas as pd

# Cleaning imports
import nltk
from nltk.corpus import stopwords


# Preprocessing imports
# from sklearn.feature_extraction.text import CountVectorizer

# Helper functions
def log(*args, **kwargs):
    """
    I'm using this to print whatever i want with [timestamp] at the beginning of each print
    :param args:
    :param kwargs:
    :return:
    """
    print(f'[{datetime.now().strftime("%H:%M:%S:%f")[:-3]}]', *args, **kwargs)


class Dataset:
    """
    Used to explore the dataset.
    TODO: Add methods & desc.
    """
    helper_labels = {
        0: 'Introverted',
        1: 'Extroverted'
    }

    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path,
                              engine='pyarrow')  # This reduces the loading time by 60%

    def __str__(self):
        return (f"The dataset contains {len(self.df)} rows, and 3 columns: {set(self.df.columns)} \n\n"
                f"The dataset has the value counts: {self.check_imbalance()}\n\n"
                f"Every row represents a user's posts and comments, every row has 1500 space-separated 'words'\n\n"
                f"There are {self.author_count()} unique authors.")

    def info(self):
        return self.df.info()

    def head(self, nrows=10):
        return self.df.head(nrows)

    def author_count(self):
        return len(self.df.author_id.unique())

    def check_imbalance(self):
        counts = self.df.extrovert.value_counts()
        labelled_counts = {self.helper_labels[0]: counts[0],
                           self.helper_labels[1]: counts[1]}
        return labelled_counts


class CleanData(Dataset):
    """
    There exists a few possible data cleaning techniques:
        i. Lowercase
        ii. Remove link
        iii. Punctuation
        iv. Stopwords
    I'm implementing these as functions where we can turn them on or off to see the effect on output of model.

    """

    def __init__(self, path,
                 remove_lowercase=True,
                 remove_punctuation=True,
                 remove_links=True,
                 remove_stopwords=True):
        super().__init__(path)
        self.remove_lowercase, self.remove_punctuation = remove_lowercase, remove_punctuation
        self.remove_links, self.remove_stopwords = remove_links, remove_stopwords

    def run(self):
        if self.remove_lowercase:
            self.lowercase()
        if self.remove_punctuation:
            self.punctuation()
        if self.remove_links:
            pass  # TODO: Implement
        if self.remove_stopwords:
            self.stopwords()

        return self.df

    def lowercase(self):
        log('Removing uppercase letters...')
        self.df.post = self.df.post.str.lower()

    def links(self):
        """
        There exists a total of 7 rows:
            5 of which start with 'https//' -> {12715, 20059, 29577, 35532, 37062}
            2 of which start with 'http//' -> {31107, 31118}
        :return: NotImplemented
        """

        return NotImplemented

    def punctuation(self):
        log('Removing punctuation...')
        self.df.post = self.df.post.str.replace(f'[{string.punctuation}]', '')

    def stopwords(self):
        try:
            log('Removing stopwords...')
            stop_words = set(stopwords.words('english'))
            if not self.remove_lowercase:
                self.df.post = self.df.post.apply(lambda x: ' '.join(
                    [word for word in x.split() if word.lower() not in stop_words]))
            else:
                self.df.post = self.df.post.apply(lambda x: ' '.join(
                    [word for word in x.split() if word not in stop_words]))
        except LookupError:
            log('nltk stopwords not found')
            log('Downloading nltk stopwords...')
            nltk.download('stopwords')
            log('Download stopwords successful, please re-run the script')



if __name__ == '__main__':
    data_path = '../data/changed_columns.csv'
    # data = ExploreDataset(path=data_path)
    # count = CleanData(data_path)
    # print(count)
    df = CleanData(path=data_path).run()
    print(df.head())
