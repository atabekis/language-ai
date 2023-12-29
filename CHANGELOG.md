### V2.4 - 05d9f77
 _Ata Bekişoğlu - 29/12/2023_
* Added `neural.py` under _methods_.
  * The class NeuralNetwork takes in model type and runs a neural model. Currently, I've only implemented a **Convolutional Neural Network**
  * Testing of the neural networks can be found under `tests/explore_neural.ipynb`
* Small changes to `process.py`,  `reader.py`, `util.py`
  * Changed the seed type variable naming
  * Some much needed clean-up
  
### V2.3 - 59d5a8c
 _Ata Bekişoğlu - 28/12/2023_
* I restructured quite a lot of the repo here since 've realized we don't really need most of the previous code after using sklearn Pipelines.
* It is quite a lot of file restructuring to list here, check [this lovely commit](https://github.com/atabekis/language-ai/commit/59d5a8c28a3bde0da0010ab0a57f3e29135ce796)
* Added the function build_pipeline() which returns a sklearn pipeline based on key given by the use- we'll use this to compare models and techniques in the paper.
* Adding some extra code to make sure theta the code is reproducible: all seeds are set to **5**

### V2.2 - 1083919
 _Ata Bekişoğlu - 22/12/2023_
* Created the notebook ata_testing.ipynb:
  * Quite a lot of findings in ths notebook actually...
  * Figured that the spacy method vs. nltk don't really differ but nltk outperforms spacy for some reason and i simply do not understand!
  * Some basic EDA on tokenized/cleaned data
  * Tfidf vectorizer + Word2Vec model using mean embeddings
  * Logistic regression on the different vecorizers
  * **IMPORTANT: found something called sklearn.pipeline** this is big news since it makes the code like 90% shorter and more readable lol.
  * Checking the SMOTE technique for imbalanced data - didn't understand a lot, goes in the backburner.
### V2.1 - f105d5b
 _Ata Bekişoğlu - 19/12/2023_
* This commit includes the vectorization and some new exploratory code
* implemented the class Vectorizer in process.py - this one is also compatible with three vectorizing engines: bow, tfidf, w2v
* renaming some Jupyter notebooks:
  * explore -> explore_data
  * some parts of explore -> explore_fastText
* created explore_methods.ipynb for testing vectorizing methods and some models, their scores etc.
* realized i still haven't put a requirements.txt -- added requirements.txt

### V2.0 - 0ee5596
 _Ata Bekişoğlu - 15/12/2023_
* Version 2, since this was a pretty big commit!
* clean.py:
  * cleaning the html attributes - decode
  * removing the lemmatize code
* tokenize.py
  * implemented a spacy pipe for tokenization using multiprocessing - best time i got so far is 6 minutes, there are some outdated test cases written down in tokenize.py
* process.py:
  * wrote the Tokenize class, this is for testing different tokenization engines such as regex, spacy and nltk - check the docymentation and the class itself, its beautiful :)
* Small changes: additions of helper functiıns to util.py and implementing absolute paths using the os library.

### V1.4 - 75b5dd6
 _Ata Bekişoğlu - 12/12/2023_
* Restructure:Jupyter notebooks are placed under /tests, cleaning and processing functions/classes are placed under /methods
* Added util.py for helper functions such as 'log()'
* Lemmatization not working as intended, implemented the parallel_lemmatize function in order tot test multiprocessing- still not working properly :(
### V1.3 - 6e9cdcb
 _Ata Bekişoğlu - 11/12/2023_
* Performing analyses using fastText - the findings are fairly skewed due to the imbalance in the dataset.
* **Restructure**: placing the jupyter notebooks in their own directory.
* Implementing classes:
  * Dataset: wrapper class ato represent the data + give insights
  * CleanData: cleans the data using class methods: check commit or docsting of CleanData for more information.
### V1.2 - 4a480ad
 _Ata Bekişoğlu - 07/12/2023_
* Changes to explore.ipynb:
  * exploring the data using spacy - tokenization and learning spacy pipes
  * spent quite some time installing fastText, some basic modelling with fastText
  * Added BACKLOG.md for keeping track of TODOs

### V1.1 - 0c78fea
 _Ata Bekişoğlu - 05/12/2023_ 
* First commit, checking the dataset through explore.ipynb

