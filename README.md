## Unveiling Personality Traits: A Comparative Study of NLP Models for Distinguishing Introverted and Extroverted Authors in Imbalanced Text Datasets
**Aadersh KalyanSundaram, Ata Bekişoğlu, Emre Sarp Kamal**\
_JBC090 - Language & AI, \
Eindhoven University of Technology, Tilburg University
2023-2024_

 > ADD: Link to the paper

### Table of Contents - add links
* [About](https://github.com/atabekis/language-ai?tab=readme-ov-file#about)
* [Data](https://github.com/atabekis/language-ai?tab=readme-ov-file#data)
  * [Including own data](https://github.com/atabekis/language-ai?tab=readme-ov-file#including-own-data)
* [Getting started](https://github.com/atabekis/language-ai?tab=readme-ov-file#getting-started)
  * [Build your own models/pipelines](https://github.com/atabekis/language-ai?tab=readme-ov-file#building-your-own-modelspipelines)
* [Reproducibility](https://github.com/atabekis/language-ai?tab=readme-ov-file#reproducibility)
* [Dependencies](https://github.com/atabekis/language-ai?tab=readme-ov-file#dependencies)


### About
This study explores the area of [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing) (NLP) 
to tackle the challenge of recognizing personality traits, specifically introversion and extroversion, from [Reddit](https://reddit.com) 
text data. We looked into various NLP techniques ranging from [Scikit-learn](https://scikit-learn.org/) to [Keras](https://keras.io/) neural networks to assess model effectiveness and to deal with class imbalance with [Imbalanced-learn](https://imbalanced-learn.org/stable/) models.

### Data
The data for the project is provided from the [_Pushshift Reddit API_](https://github.com/pushshift/api), we do not provide the data publicly on this repository. \
Our dataset includes posts and comments from Reddit users, with binary labels on _introvertendess_ & _extrovertedness_.
#### Including own data:
User data can be added under the `data/` directory, unfortunately, this project will only work with datasets similar to our base dataset. In future revisions, plans are made to implement features that will be able to work on most dataset provided by the user. Additionally, our current models will only work with binary labels, such as '1' or '0'.


### Getting started
Cloning the repository:
```shell
git clone https://github.com/atabekis/language-ai.git
```


We recommend using Python 3.9 or above and with a code editor such as [PyCharm](https://www.jetbrains.com/pycharm/) or [VSCode](https://code.visualstudio.com/).
1. Installing the required packages

```shell
 pip install -r requirements.txt
```

> **Notice:** the pip installer for the [FastText](https://fasttext.cc/) library **might** return errors when installing through pip on Windows systems. 
> We recommend building the library locally on the preferred environment.
> > Please refer to [this guide](https://github.com/facebookresearch/fastText/issues/1343#issuecomment-1646580169) on how to.

2. Due to time constraints, no argument parser is implemented within the project. All methods can be run thorugh `main.py` and its parameters.
```python
def main(
        # Main experiments
        single_experiment: str = None,
        multiple_experiments: bool = True,
        cross_validate_experiments: bool = False,
        # Controls for the experiment class
        time_experiments: bool = True,
        verbose: bool = True,
        debug: bool = False,
        # Saving and loading models
        load_existing_models: bool = False,
        save_models: bool = True):
```
Explanation of the code block above:
* `single_experiment`: pass any of the six models implemented within the project:
  * `'naive-bayes', 'svm', 'logistic', 'fasttext', 'cnn', 'lstm'`
* `multiple_experiments`: Run all the six models.
* `cross_validate_experiments`: Running the 3 sklearn classifiers with k-fold cross-validation.

> The outputs/metrics of these experiments are saved under `methods/output`.

3. The final bit of evaluation is performed under `evaluate.ipynb`, where we visualize the decision factors behind the models.

#### Building your own models/pipelines
Under `methods/process.py` the `build_pipelines()` function can be found. It follows the structure:
```python
models = {
        # Naive Bayes Model wih Bag of Words
        'naive-bayes': [
            ('vectorizer', CountVectorizer(ngram_range=(1, 2), binary=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', MultinomialNB())
        ],
        # Support Vector Machines with tf*idf
        'svm': [
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True)),
            ('resampler', resampler(model=resampling_method)),
            ('classifier', LinearSVC(dual='auto'))
        ],
# ------ More models ------- #
```
The `resampler` method can also be found under the same document, one of the following methods can be passed onto the function:
* `'random-over', 'random-under', 'smote', 'adasyn', 'tomek'`

User-defined pipelines can be added to this function in a similar format. Example:
```python
'mlp-classifier': [
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,2))
]
```

More documentation about the methods implemented can be found under `methods/README.md`


### Reproducibility
In order to recreate the results presented in the paper, we put an emphasis on reproducability within the code.
All models with random decision factors are passed the global `__RANDOM_SEED__` variable which is defined in `config.py`.
The default value of this variable is set to `5`.

### Dependencies
We recommend the installation of these packages to ensure an end-to-end process with no errors.
```python
# Base packages
numpy~=1.24.3
pandas~=2.1.2
joblib~=1.3.2  # To save and load pipelines
scikit-learn~=1.3.2

pyarrow~=14.0.1  # This package is used to make pandas faster at reading large files.

nltk~=3.8.1  # Preprocessing
keras~=3.0.2  # Neural Networks
fasttext~=0.9.2  # Word Embeddings / standalone model
tensorflow~=2.13.1  # Neural Networks
imbalanced-learn~=0.11.0  # Resampling

# Only used in some Jupyter Notebooks
tqdm~=4.66.1  # Progress bar
gensim~=4.3.2  # Word Embeddings
```

