### V3.3 - ADD
 _Ata Bekişoğlu - 05/01/2024_
* Once again, quite some changes:
* Got rid of `tune_models.py` since we really don't have enough time and it takes quite some time to process - i ave no time to debug. 
* Also, got rid of `evaluate.py`, im keeping the `evaluate.ipynb` since i can have a flow and show some visualizations on the side. This file does not contribute at all to the experiment but its using the experiment class to extract some findings.
* Implemented `cross_validate_experiments()` inside `Experiment`, which cross validates experiments :)
  * I'm able to run the sklearn models but the neural networks will take quite some time if i try to cross-validate. I'll decide on whether to CV the neural networks or not after i keep it running for a while.
* Inside `explore_sampling.ipynb` i also added a loop to run LinearSVC on all resampling methods -- I did this a few commits ago but maybe we'll use the table this cell exports in the paper.
* `Reader`: changed function name `remove_lowercase` → `remove_uppercase`
* `Experiment`: added abort save so that we don't loose the calculated metrics if one of the neural networks is acting up.
* `Expreiment`: Also added debugging where i just cut the data to see if a model/method is working.
* Some cleanup and commenting/documenting

### V3.2 - 769bdfc

 _Ata Bekişoğlu - 03/01/2024_ 

* It was indeed not the end of the coding part... Some big changes:
* Changes to `process.py`:
  * Added the Word2Vec model that is built in the new `models.py` file. I still have some issues making it work with the resampling methods.
  * Added the method to save the pipelines `save_pipe=True` that saves the pipes under the _pipelines_ directory
* Added the `evaluate.py` and `evaluate.ipynb`, for now they're the same thing. And i have to decide on which one to keep in the future.
  * This is used to extract information about the coefficients and selection process of the sklearn classifiers. Some very nice findings found here → im going to list them in a README somewhere.
* Added the `tune_models.py` file with the class `Tuner`:
  * Uses k-fold cross validation on the pipelines to extract the best hyperparameters for the models.


### V3.1 - e1c2c75
 _Ata Bekişoğlu - 31/12/2023_ 
* Happy new years! I'm done with coding, and this was the end of the coding part for the project. It truly is a new years miracle.
* Most changes were done in `process.py`:
  * added the function `resampler()`which returns a resampling method from the imblearn library.
  * this then is used by the newly modified `build_pipeline()` to include resamplers.
* Implemented the class `Experiment`
  * Final step of the project; reads the data using `Reader`, and passes the data into the pipelines.
  * In `perform_single_experiment()` we select one model and perform it
  * In `perform_mant_experiments()` all models available in the `__init__` method are executed.
  * The metrics of the models are kept and then exported into a latex table
* Added `config.py` to keep the absolute paths to the data and export directories.
* Small changes to the notebooks, requirements.txt, and some much needed documentation to functions and classes in `reader.py`.


### V3.0 - a9118ad
 _Ata Bekişoğlu - 30/12/2023_ 
* Hooray! Version 3!
* Changes to `neural.py`:
  * Implemented **Long Short-Term Memory** (LSTM) model under `NeuralNetwork` class.
  * Changes to the fit method to include more controlled epochs and early stopping.
  * Implemented the `predict()` method.
* Changes to `process.py` → Added neural pipelines.
* Big changes to `reader.py`:
  * Changed the `Dataset` and `CleanData` classes to read the dataframe instead of the raw csv.
  * Connected the aforementioned classes to `Reader` in order to have an end-to-end process.
* Started on the `methods/README.md` for better understanding!
* TODO: add class method to save (all) classifiers/networks, so we don't wait as long in the future!

_Emre Sarp Kamal - 30/12/2023_
* Done with working towards the imbalanced dataset problem:
  * Worked in a separate notebook called SMOTE_test (implement this as .py file to PyCharm later and change name) as PyCharm kept giving errors in every possible step:
  * Researched quite a bit and implemented SMOTE, did not work out as intended. Didn't see the expected results, started looking into other methods.
  * Worked with `stratify` in train_test_split, seemed to work out better
  * Implemented TfidfVectorizer as a simple tokenizer and used Naive Bayes as classifier to compare measurements with and without stratify
  * Found out about `StratifiedShuffleSplit` after research, also used it with the same tokenizer and classifier for comparison
  * Only thing left is to implement these to our project to talk in the paper
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

_Emre Sarp Kamal - 25/12/2023_
* After some previous research on fastText created a separate branch emretest:
  * created emre_testing by copying ata_testing to make separate changes
  * worked on fastText kept getting errors trying to install it
  * data is imbalanced will work towards finding a solution to this instead, working on emre_testing looking into SMOTE
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

_Emre Sarp Kamal - 07/12/2023_
* Learning and trying out spaCy, did tokenization on a subset of the data in a separate notebook
### V1.1 - 0c78fea
 _Ata Bekişoğlu - 05/12/2023_ 
* First commit, checking the dataset through explore.ipynb

_Emre Sarp Kamal - 06/12/2023_
* Loaded dataset and did exploration
