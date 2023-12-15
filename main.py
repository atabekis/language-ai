"""
Copyright (C) 2023-2024 Eindhoven University of Technology & Tilburg University
JBC090 - Language & AI

Aadersh Kalyanasundaram, Ata Bekişoğlu, Egehan Kabacaoğlu, Emre Kamal

If the code doesn't run when cloning from GitHub, the data folder is assumed to be missing.
"""
from methods.process import Tokenizer


if __name__ == '__main__':
    token = Tokenizer(
        path='data/cleaned_extrovert.csv',
        engine='spacy',
        save_csv=True
    )

    token.run()
