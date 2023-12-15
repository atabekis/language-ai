"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""

# Python imports
import os

# Local imports
from methods.process import Tokenizer
from methods.clean import CleanData

__FILE_PATH__ = os.path.join(os.path.dirname(__file__), 'data', 'extrovert_introvert.csv')


def clean_and_tokenize():
    clean = CleanData(__FILE_PATH__)
    clean.run()

    clean_file_path = clean.out_path()

    #  Tokenize
    token = Tokenizer(clean_file_path)
    token.run()


if __name__ == '__main__':
    clean_and_tokenize()
