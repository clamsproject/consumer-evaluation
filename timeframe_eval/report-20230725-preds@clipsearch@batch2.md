#### app
The clipsearch app was used to generate the preds MMIF files (located [here](https://github.com/clamsproject/aapb-evaluations/tree/750fa21c415c16b780c2c0e2fb3e6f031ab88511/timeframe-eval/preds%40clipsearch%40batch2)).

#### evaluation dataset
The gold annotation can be found [here](https://github.com/clamsproject/aapb-annotations/blob/61bd60e99ef24a1ca369e23de8b2c74bb2cb37d3/newshour-chyron/golds/batch2/2022-jul-chyron.csv).

#### evaluation code
The script used for evaluation is [here](https://github.com/clamsproject/aapb-evaluations/blob/4c96f78f8b03d6ad9660810673989973620a6650/timeframe-eval/evaluate.py).

#### evaluation metrics
This batch was evaluated by treating chyron detection as a binary classification task and recording the precision and recall scores.
That is, for each frame in a file, there are four possibilities:

1. the frame contains a chyron and the app detects a chyron (true positive)
2. the frame does not contain a chyron but the app detects a chyron (false positive)
3. the frame contains a chyron but the app does not detect a chyron (false negative)
4. the frame does not contain a chyron and the app does not detect a chyron (true negative)
   
Precision is a measure of the number of correctly identified chyron timeframes as compared to the number of identified chyron timeframes (i.e. true positives / (true positives + false positives)), and recall is a measure of the number of correctly identified chyron timeframes as compared to the number of actual chyron timeframes (i.e. true positives / (true positives + false negatives)). See the [pyannote.metrics documentation](https://pyannote.github.io/pyannote-metrics/index.html) for more details.

#### evaluation results
The evaluation results for this batch were as follows (where F1 is the harmonic mean of the precision and recall):

* Precision = 0.0  
* Recall = 0.0  
* F1 = 0.0  

Some reasons for the low scores include small batch size and the inherent difficult of the chyron task.

#### side-by-side views
The side-by-side views of the gold annotations and app predictions can be found [here](https://github.com/clamsproject/aapb-evaluations/tree/4c96f78f8b03d6ad9660810673989973620a6650/timeframe-eval/results%40clipsearch%40batch2).
Each file contains rows of time intervals, gold annotations, and app predictions, in that order. For example, a row of

> (4 - 5), 1, 1  

means that in the interval of 4 seconds to 5 seconds, the gold annotations record a chyron, and the app also predicts a chyron, whereas a row of

> (11 - 12), 0, 1  

means that in the interval of 11 seconds to 12 seconds, the gold annotations do not record a chyron, but the app does predict a chyron.
