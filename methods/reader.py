# Core imports
import os
import html
import string
import pandas as pd

# Cleaning imports
import nltk
from nltk.corpus import stopwords

# Manipulate the dataset
from sklearn.model_selection import train_test_split

# Local imports
from util import log, save_file_to_path
from config import __PROJECT_PATH__

# Reproducibility
__RANDOM_SEED__ = 5


class Dataset:
    """
    Represents the dataset, performs basic EDA functions
    :param dataframe: pandas dataframe
    """
    helper_labels = {
        0: 'Introverted',
        1: 'Extroverted'
    }

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.change_column_names()

    def __str__(self):
        return (f"The dataset contains {len(self.df)} rows, and 3 columns: {set(self.df.columns)} \n\n"
                f"The dataset has the value counts: {self.check_imbalance()}\n\n"
                f"Every row represents a user's posts and comments, every row has 1500 space-separated 'words'\n\n"
                f"There are {self.author_count()} unique authors.")

    def read_data(self):
        """
        Deprecated.
        :return: None
        """
        # return pd.read_csv(self.path, engine='pyarrow')  # This reduces the loading time by 60%
        return DeprecationWarning

    def info(self) -> None:
        """Info function call from pandas"""
        return self.df.info()

    def head(self, nrows: int = 10) -> pd.DataFrame:
        """
        returns first n rows of the dataframe
        """
        return self.df.head(nrows)

    def author_count(self) -> int:
        """Total count of the authors in the dataframe"""
        return len(self.df.author_id.unique())

    def check_imbalance(self) -> dict[int, str]:
        """Checks the imbalance of the labels/targets"""
        counts = self.df.label.value_counts()
        labelled_counts = {self.helper_labels[0]: counts[0],
                           self.helper_labels[1]: counts[1]}
        return labelled_counts

    def change_column_names(self) -> None:
        """Changes the misspelled column names in the original dataframe"""
        self.df.rename(columns={'auhtor_ID': 'author_id', 'extrovert': 'label'}, inplace=True)

    def label_metrics(self, save_latex: bool = True) -> pd.DataFrame:
        """Returns basic information about the two labels in the dataframe
            This method is used in the paper under the Data section as a table
        """
        avg_metrics_per_label = self.df.groupby('label')['post'].apply(lambda x: {
            'Avg. Word Length': x.apply(lambda post: len(post.split())).mean(),
            'Avg. Char. Count': x.apply(len).mean(),
            'Normalized Vocab. Size': len(set(' '.join(x).split())) / x.shape[0],
            'Avg. # Unique Words': x.apply(lambda post: len(set(post.split()))).mean(),
            'Total': x.shape[0]
        }).round(1)
        transposed_df = avg_metrics_per_label.transpose()

        if save_latex:
            transposed_df.to_latex(os.path.join(__PROJECT_PATH__, 'methods', 'output', 'eda_raw_data.tex'),
                                   index=True, escape=False)

        return transposed_df


class CleanData(Dataset):
    """
    This class is used to clean the Dataset object.
    Use the function run() in order to clean the dataset.

    :param dataframe:
        Pandas dataframe
    :param remove_uppercase: bool, optional, default True
        Whether to remove the uppercase characters
    :param remove_punctuation: bool, optional, default True
        Whether to remove punctuation characters
    :param remove_links: bool, optional, default True
        Whether to remove hyperlinks
    :param remove_stopwords: bool, optional, default True
        Whether to remove stopwords
    :param lemmatize_words: bool, optional, default True
        Lemmatize the words
    :param save_csv: bool, optional, default False
        Save the csv under the data directory
    """

    def __init__(
            self,  # Methods marked with '*' are the ones used in the paper.
            dataframe: pd.DataFrame,
            remove_uppercase=True,  # *
            remove_punctuation=True,  # * Also decodes the HTML attributes -we count them as punctuation
            remove_links=False,
            remove_stopwords=True,  # *
            lemmatize_words=False,
            save_csv=False):

        super().__init__(dataframe)
        self.remove_uppercase, self.remove_punctuation = remove_uppercase, remove_punctuation
        self.remove_links, self.remove_stopwords = remove_links, remove_stopwords
        self.lemmatize_words = lemmatize_words
        self.save_csv = save_csv
        if save_csv:
            self.out_path = save_file_to_path('data', 'cleaned_extrovert.csv')

    def run(self) -> pd.DataFrame:
        """Main function to clean the data based on the class parameters"""
        if self.remove_uppercase:
            self.lowercase()
        if self.remove_punctuation:
            self.punctuation()
            self.decode()

        if self.remove_links:
            pass  # Does not affect the analysis, thus not implemented here for simplicity
        if self.remove_stopwords:
            self.stopwords()
        if self.lemmatize_words:
            self.lemmatize()
        if self.save_csv:
            self.save()
        return self.df

    def lowercase(self) -> None:
        """Normalize the text"""
        log('[Clean] Removing uppercase letters...')
        self.df.post = self.df.post.str.lower()

    def decode(self) -> None:
        """Decode the html attributes such as escape chars."""
        log('[Clean] Decoding HTML attributes...')
        self.df.post = self.df.post.apply(lambda x: html.unescape(x))

    def links(self) -> NotImplemented:
        """
        Remove hyperlinks from text
        There exists a total of 7 rows:
            5 of which start with 'https//' -> {12715, 20059, 29577, 35532, 37062}
            2 of which start with 'http//' -> {31107, 31118}
        :return: NotImplemented
        """

        return NotImplemented

    def punctuation(self) -> None:
        """Removes punctuation from the text"""
        log('[Clean] Removing punctuation...')
        self.df.post = self.df.post.str.replace(f'[{string.punctuation}]', '')
        self.df.post = self.df.post.str.replace(f'\\', '')

    def stopwords(self) -> None:
        """Remove stopwords using nltk"""
        try:
            log('[Clean] Removing stopwords...')
            stop_words = set(stopwords.words('english'))
            if not self.remove_uppercase:
                self.df.post = self.df.post.apply(lambda x: ' '.join(
                    [word for word in x.split() if word.lower() not in stop_words]))
            else:
                self.df.post = self.df.post.apply(lambda x: ' '.join(
                    [word for word in x.split() if word not in stop_words]))
        except LookupError:
            log('[Clean] nltk stopwords not found')
            log('[Clean] Downloading nltk stopwords...')
            nltk.download('stopwords')
            log('[Clean] Download stopwords successful, please re-run the script')

    def lemmatize(self) -> NotImplemented:
        """Lemmatize the text, not implemented yet"""
        return NotImplemented
        # log('Lemmaizing text...')
        # log('This will take a while...')
        # self.df.post = parallel_lemmatize(self.df.post)

    def save(self) -> None:
        """if save=True, saves the csv under data/cleaned_extrover.csv"""
        log('[Clean] Saving cleaned data...')
        self.df.to_csv(self.out_path, index=False)


class Reader:
    """
    Main wrapper around the dataset
    :param path: str,
        The path to the uncleaned dataset (csv)
    :param clean: bool, optional, default=True
        If true calls CleanData class, for more information check CleanData
    :param: split: bool, optional, default=True
        Splits the dataframe using sklean train test split
    :param show_info: bool, optional, default=True
        Shows information about the dataset such as: value counts, imbalance and basic properties
    """

    def __init__(
            self,
            path: str,
            clean: bool = True,
            split: bool = True,
            show_info: bool = True):

        self.df = pd.read_csv(path, engine='pyarrow')
        self.df = Dataset(self.df).df

        self.labels = self.df['label']
        self.posts = self.df['post']

        self.train = [[], []]  # X_train, y_train
        self.test = [[], []]  # X_test, y_test

        if show_info:
            self.info()
        if clean:
            self._clean_data()
        if split:
            self._split_data()

    def info(self) -> str:
        """Instance of Dataset class, represents basic info about the dataset"""
        return Dataset(self.df).__str__()

    def _clean_data(self) -> pd.DataFrame:
        """Calls the CleanData class and cleans the dataframe"""
        log('[Clean] Cleaning data...')
        self.df = CleanData(self.df).run()
        return self.df

    def _split_data(self) -> None:
        """split the dataframe using train_test_split"""
        log('[Reader] Splitting the dataframe into train/test sets...')
        self.train[0], self.test[0], self.train[1], self.test[1] = train_test_split(
            self.df['post'], self.df['label'],
            test_size=0.2, random_state=__RANDOM_SEED__)


if __name__ == '__main__':
    df = Reader('../data/extrovert_introvert.csv').df
    dataset = Dataset(df)
    dataset.label_metrics()
