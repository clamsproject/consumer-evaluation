#### app
The slatedetection app was used to generate the preds MMIF files (located [here](https://github.com/clamsproject/aapb-evaluations/tree/23d04ddb5745c24cce642fa5ce9728ea37132af2/timeframe-eval/preds%40slatedetection%40aapb-collaboration-7)).  

#### evaluation dataset
The gold annotation can be found [here](https://github.com/clamsproject/aapb-annotations/blob/62e4d399c3ba1ea47719d504d0f088a768486177/january-slates/230101-aapb-collaboration-7/CLAMS_slate_annotation_metadata.csv).  
  
#### evaluation code
The script used for evaluation is [here](https://github.com/clamsproject/aapb-evaluations/blob/4c96f78f8b03d6ad9660810673989973620a6650/timeframe-eval/evaluate.py).  
  
#### evaluation metrics
This batch was evaluated by treating slate detection as a binary classification task and recording the precision and recall scores.  
That is, for each frame in a file, there are four possibilities:   

1. the frame contains a slate and the app detects a slate (true positive)
2. the frame does not contain a slate but the app detects a slate (false positive)
3. the frame contains a slate but the app does not detect a slate (false negative)
4. the frame does not contain a slate and the app does not detect a slate (true negative)    

Precision is a measure of the number of correctly identified slate timeframes as compared to the number of identified slate timeframes (i.e. true positives / (true positives + false positives)), and recall is a measure of the number of correctly identified slate timeframes as compared to the number of actual slate timeframes (i.e. true positives / (true positives + false negatives)).
See the [pyannote.metrics documentation](https://pyannote.github.io/pyannote-metrics/index.html) for more details.  
    

#### evaluation results
The evaluation results for this batch were as follows (where F1 is the harmonic mean of the precision and recall):  

* Precision = 0.9036751247089342  
* Recall = 0.8353046960857986  
* F1 = 0.8681458707902895


#### side-by-side views
The side-by-side views of the gold annotations and app predictions can be found [here](https://github.com/clamsproject/aapb-evaluations/tree/4c96f78f8b03d6ad9660810673989973620a6650/timeframe-eval/results%40slatedetection%40aapb-collaboration-7).  
Each file contains rows of time intervals, gold annotations, and app predictions, in that order. For example, a row of  
> (4 - 5), 1, 1  

means that in the interval of 4 seconds to 5 seconds, the gold annotations record a slate, and the app also predicts a slate, whereas a row of
> (11 - 12), 0, 1  

means that in the interval of 11 seconds to 12 seconds, the gold annotations do not record a slate, but the app does predict a slate.
