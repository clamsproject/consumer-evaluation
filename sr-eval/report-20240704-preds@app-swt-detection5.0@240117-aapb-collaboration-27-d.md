# Scenes With Text Evaluation Report


## Report Instance of Evaluation Information
* 2024-07-03
* [app-SWT-detection](https://github.com/clamsproject/app-swt-detection/tree/6b12498fc596327ec47933b7f785044da2f8cf2f), version 5.0
* [scene-recognition/golds](https://github.com/clamsproject/aapb-annotations/tree/9cbe41aa124da73a0158bfc0b4dbf8bafe6d460d/scene-recognition/golds).
* [preds@app-swt-detection5.0@240117-aapb-collaboration-27-d](https://github.com/clamsproject/aapb-evaluations/tree/4bfcf3250700567aae352a710d5cdb6bc1fcdca4/sr-eval/preds%40app-swt-detection5.0%40240117-aapb-collaboration-27-d).
* [sr-eval/evaluation.py](https://github.com/clamsproject/aapb-evaluations/blob/4bfcf3250700567aae352a710d5cdb6bc1fcdca4/sr-eval/evaluate.py).
* `python evaluate.py -m /path/to/repo/aapb-evaluations/sr-eval/preds@app-swt-detection5.0@240117-aapb-collaboration-27-d`

## Metrics
This dataset was evaluated using the typical `Precision, Recall, F1` scores for classification. Specifically, each
predicted timepoint label is compared to the gold standard label annotation. Each document 
(in this case the single `cpb-aacip-259-wh2dcb8p.csv`) has a set of macro-averaged scores across all labels, and then the
`dataset_scores.csv` contains the total (micro-averaged across the entire dataset) scores per label, and a final overall
(micro) averaged set of scores (the `all` column)
Note that labels which do not appear (on a per-document basis or otherwise) are not included in these averages.

## Results
`cpb-aacip-259-wh2dcb8p.csv`

| label   |precision         |recall|f1                |
|---------|------------------|------|------------------|
| -       |0.8874425727411945|0.8827113480578828|0.8850706376479572|
| B       |1.0               |1.0   |1.0               |
| S       |1.0               |0.8571428571428571|0.923076923076923 |
| O       |0.0               |0.0   |0.0               |
| L       |1.0               |0.3333333333333333|0.5               |
| M       |0.32142857142857145|0.75  |0.45000000000000007|
| G       |0.390625          |0.5208333333333334|0.44642857142857145|
| Y       |0.0               |0.0   |0.0               |
| F       |0.6818181818181818|0.45918367346938777|0.548780487804878 |
| P       |0.5296610169491526|0.5605381165919282|0.5446623093681917|
| T       |0.0               |0.0   |0.0               |
| I       |1.0               |0.7142857142857143|0.8333333333333333|
| E       |0.0               |0.0   |0.0               |
| C       |0.5306122448979592|0.65  |0.5842696629213483|
| R       |0.0               |0.0   |0.0               |
| average |0.4894391725223373|0.4485352250809625|0.44770812837208024|

`dataset_scores.csv` has the same values for each label, as in this case there is only one documnent. However, the
overall micro-averaged set of scores (from the `all` row) is as follows:

| label |precision         |recall|f1                |
|-------|------------------|------|------------------|
| all   |0.7916666666666666|0.7916666666666666|0.7916666666666666|


## Limitations/Issues
* This script seems to be somewhat inefficient in terms of runtime. This may be due to not utilizing various evaluation
libraries (like numpy), and it might be worthwhile in the future to refactor the script such that those libraries are
used.
