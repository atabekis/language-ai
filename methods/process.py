"""
This file is used to process the data.
This includes the steps
    1. Tokenization: How?????
    2. Vectorization: How???

    Additionally: inbetween -> Move lemmatization here since we can use tokenized sentences.
"""

# Python imports
import numpy as np
import pandas as pd


class Tokenizer:
    def __init__(self, path):
        self.path = path

    def read_data(self):
        return NotImplemented
