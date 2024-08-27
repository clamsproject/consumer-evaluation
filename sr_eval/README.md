# Scene Recognition Evaluation
This involves evaluating the results of the scenes-with-text classification task. While SWT returns both timepoint and
timeframe annotations, this subdirectory is focused on timepoints.
The goal is to have a simple way of comparing different results from SWT. 

# Required Input
To run this evaluation script, you need the following:

* Set of predictions in MMIF format (either from the preds folder in this repo
or generated from the [SWT app](https://github.com/clamsproject/app-swt-detection/tree/6b12498fc596327ec47933b7f785044da2f8cf2f)
* Set of golds in csv format (either downloaded from the annotations repository
using goldretriever.py, or your own set that exactly matches the format present in [aapb-annotations](https://github.com/clamsproject/aapb-annotations/tree/9cbe41aa124da73a0158bfc0b4dbf8bafe6d460d/scene-recognition/golds)

There are three arguments when running the script: `-mmif-dir`, `-gold-dir`, and `count-subtypes`.
The first two are directories that contain the predictions and golds, respectively. The third is a boolean value that
determines if the evaluation takes into account subtype labels or not.
* Our standard for naming prediction (mmif) directories is as follows:
* `preds@app-swt-detection<VERSION-NUMBER>@<BATCH-NAME>`.

Note that only the first one is required, as `-gold-dir` defaults to the set of golds downloaded (using `goldretriever`)
from the [aapb-annotations](https://github.com/clamsproject/aapb-annotations/tree/9cbe41aa124da73a0158bfc0b4dbf8bafe6d460d/scene-recognition/golds) repo,
and `count-subtypes` defaults to `False`.

# Usage
To run the evaluation, run the following in the `sr-eval` directory:
```
python evaluate.py --mmif-dir <pred_directory> --gold-dir <gold_directory> --count-subtypes True
```

# Output Format
Currently, the evaluation script produces a set of `{guid}.csv` files for each document in the set of predictions, and a
`dataset-scores.csv`.
* `{guid}.csv` has the label scores for a given document, including a macro-average of label scores.
* `dataset-scores.csv` has the total label scores across the dataset, including a final micro-average of all labels.

These contain the precision, recall, and f1 scores by label. In each document, the first row 
is the negative label `-`, and specifically the `dataset-scores` has the `all` label as its second row which
represents the final micro-average of all the labels.

The output files are placed in a directory whose name is derived from the final portion (split on `@`)of the basename 
for the given prediction directory. Using our format described in [Required Input](#required-input), this would result
in the name being `scores@<BATCH-NAME>`.
