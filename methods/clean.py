# Core imports
import html
import string
import pandas as pd

# Cleaning imports
import nltk
from nltk.corpus import stopwords

# Local imports
from util import log, parallel_lemmatize


class Dataset:
    """
    Used to explore the dataset.
    TODO: Add methods & desc.
    """
    helper_labels = {
        0: 'Introverted',
        1: 'Extroverted'
    }

    def __init__(self, path: str):
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
    This class is used to clean the Dataset object.
    Use the function run() in order to clean the dataset.

    """

    def __init__(self, path: str,
                 remove_lowercase=True,
                 remove_punctuation=True,
                 remove_links=True,
                 remove_stopwords=True,
                 lemmatize_words=True,
                 save_csv=True,
                 ):
        """
        :param path: Points to the string path of the CSV file.
        :param remove_lowercase: Whether to remove the lowercase characters
        :param remove_punctuation: Whether to remove punctuation characters
        :param remove_links: Whether to remove hyperlinks
        :param remove_stopwords: Whether to remove stopwords
        :param lemmatize_words: Lemmatize the words
        """

        super().__init__(path)
        self.remove_lowercase, self.remove_punctuation = remove_lowercase, remove_punctuation
        self.remove_links, self.remove_stopwords = remove_links, remove_stopwords
        self.lemmatize_words = lemmatize_words
        self.save_csv = save_csv

    def run(self) -> pd.DataFrame:
        if self.remove_lowercase:
            self.lowercase()
        if self.remove_punctuation:
            self.punctuation()
            self.decode()

        if self.remove_links:
            pass  # TODO: Implement
        if self.remove_stopwords:
            self.stopwords()
        if self.lemmatize_words:
            self.lemmatize()
        if self.save_csv:
            self.save()
        return self.df

    def lowercase(self):
        log('Removing uppercase letters...')
        self.df.post = self.df.post.str.lower()

    def decode(self):
        log('Decoding HTML attributes...')
        self.df.post = self.df.post.apply(lambda x: html.unescape(x))

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
        self.df.post = self.df.post.str.replace(f'\\', '')

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

    def lemmatize(self):
        return NotImplemented
        # log('Lemmaizing text...')
        # log('This will take a while...')
        # self.df.post = parallel_lemmatize(self.df.post)

    def save(self):
        log('Saving cleaned data...')
        out_path = '/'.join(self.path.split('/')[:-1]) + '/cleaned_extrovert.csv'
        self.df.to_csv(out_path, index=False)


if __name__ == '__main__':
    df = CleanData(path='../data/changed_columns.csv',
                   remove_lowercase=True,
                   remove_stopwords=True,
                   remove_punctuation=True,
                   lemmatize_words=True,
                   save_csv=True,
                   ).run()
    print(df.head())
