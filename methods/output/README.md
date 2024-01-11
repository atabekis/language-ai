> ### methods/output

In this directory the output of the experiments are stored in the `.tex` format. \
Run `main.py` in order to generate the files here:

### Files here:
#### `eda_raw_data.tex`:
* Stores the output of `Dataset.label_metrics()`
* Which includes: 
  * Average Word Length,
  * Average Character Count,
  * Normalized Vocabulary Size,
  * Average Number of Unique Words,
  * Total

#### `many_experiments.tex`:
* Label metrics for the 6 implemented models

#### `many_experiments_CV.tex`:
* Label metrics for the 3 simple models, cross validated.

#### `models_top_features.tex`:
* Extracted top words from the 4 models.
