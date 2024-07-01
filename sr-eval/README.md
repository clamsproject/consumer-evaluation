# Scene Recognition Evaluation
This involves evaluating the results of the scenes-with-text classification task.
The goal is to have a simple way of comparing different results from SWT.

# Required Input
To run this evaluation script, you need the following:

* Set of predictions in MMIF format (either from the preds folder in this repo
or generated from the [SWT app](https://github.com/clamsproject/app-swt-detection) )
* Set of golds in csv format (either downloaded from the annotations repository
using goldretriever.py, or your own set that exactly matches the format present in [aapb-annotations](https://github.com/clamsproject/aapb-annotations/tree/main/scene-recognition/golds))

There are three arguments when running the script: `-mmif-dir`, `-gold-dir`, and `count-subtypes`.
The first two are directories that contain the predictions and golds, respectively. The third is a boolean value that
determines if the evaluation takes into account subtype labels or not.
Note that only the first one is required, as `-gold-dir` defaults to the set of golds downloaded (using `goldretriever`)
from the [aapb-annotations](https://github.com/clamsproject/aapb-annotations/tree/main/scene-recognition/golds) repo,
and `count-subtypes` defaults to `False`.

# Usage
To run the evaluation, run the following in the `sr-eval` directory:
```
python evaluate.py -mmif-dir <pred_directory> -gold-dir <gold_directory> -count-subtypes True
```

# Output Format
Currently, the evaluation script produces two output files: `document-scores.csv` and `dataset-scores.csv`
* `document-scores.csv` has the label scores by document, including a macro-average of label scores.
* `dataset-scores.csv` has the total label scores across the dataset, including micro-averaged results.

These contain the precision, recall, and f1 scores by label. At the moment, the scores themselves are outputted in a
dictionary format, but this is subject to change.

# Notes
As mentioned previously, this is the first version of this evaluation script and some things are subject to change
including output format and location.
