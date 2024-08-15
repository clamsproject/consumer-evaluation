# Evaluation on CLAMS RFB app

## Introduction
Recall from [RFB page](https://github.com/clamsproject/app-role-filler-binder), the app performs NER-like task to annotate _**role**_ and _**filler**_ entities and "bind" them as (key, value) pair. Therefore, our evaluation mainly assesses three annotations run by RFB: _**role**_, _**filler**_, _**binding**_.

## Metric
Our primary metric is Intersection Over Union (IOU), which is a simple but representative metric used in *Object Detection* tasks. The corresponding terms of IOU udner RFB context are:

For entities _**role**_, _**filler**_:
* Intersection: Overlap annotated strings between gold and prediction within a span of same frames
* Union: All annotated strings between gold and prediction within a span of same frames.

For the entity _**binding**_:
* Intersection: The exact match for both _**role**_ and _**filler**_.
* Union: The universe of all (_**role**_, _**filler**_) pairs

## Data
* Gold: [`aapb-annotations-44`](https://github.com/clamsproject/aapb-annotations/tree/89-rfb-gold/role-filler-binding)

## Usage
### Set up python environment
```pip install -r requirements.txt```

### Run evaluation script
```
python evaluate.py \
-- preds <preds_dir> \
-- debug \
-- in_seq
```

The script intends to provide two modes of running: sequential or parallel. 

>[!IMPORTANT] 
    As of 8/15/2024, the parallel mode hasn't been tested. So make sure include `in_seq` flag while running.