# Automatic Speech Recognition Evaluation
This involves evaluating the results of the Automatic Speech Recognition apps.

# Required Input
To run this evaluation script, you need the following:

* Set of predictions in MMIF format
* Set of golds in txt format. The gold fles must either downloaded from the annotations repository
using goldretriever.py, or your own set of files. 

There are two arguments when running the script: `-m` and `-g`.
They are directories that contain the predictions and golds, respectively. 
Note that only the first one is required, as `-gold-dir` defaults to the set of golds downloaded 
from the https://github.com/clamsproject/aapb-collaboration/tree/89b8b123abbd4a9a67c525cc480173b52e0d05f0/21 using `goldretriever`.

# Usage
To run the evaluation, run the following in the `asr-eval` directory:
```
python evaluate.py -m <pred_directory> -g <gold_directory>
```

# Output Format
Currently, the evaluation script produces one output files: `results.json`.
* `results.json` is a json file the contain the result of WER Calculation for each pair of pred/gold files.
* The WER Calculation is done by `WordErrorRate` from `torchmetrics` library.
* The result file stores the WER results depending on the evaluation conditions (currently two conditions: case-sensitive and non case-sensitive). In the future more conditions will be taken into consieration, so the result .json could extend accordingly.

A sample results.json file:
```
{"cpb-aacip-123-1234567890": [{"wer_result": 0.230140820145607, "exact_case": true}, {"wer_result": 0.2058475762605667, "exact_case": false}]}
```
