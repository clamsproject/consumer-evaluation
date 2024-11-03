# RFB Evaluation report 

## Introduction
Recall from [RFB page](https://github.com/clamsproject/app-role-filler-binder), the app performs NER-like task using finetuned BERT to annotate _**role**_ and _**filler**_ entities and "bind" them as (key, value) pair (called _**binding**_) on AAPB data. Therefore, our evaluation mainly assesses three annotations run by RFB: _**role**_, _**filler**_, _**binding**_.

## Metric
Our primary metric is Intersection Over Union (IOU), which is a simple but representative metric used in *Object Detection* tasks. However, unlike images to be evaluated in the original IOU context, RFB outputs plain texts so the logistics of evaluation needs to be adapted accordingly. 

Specifically, we define parts how "Intersection" and "Union" for entities _**role**_, _**filler**_ and _**binding**_ would be defined:

### Intersection
* Role: the entity is regarded as a set in both gold and predition, therefore, the intersection is simply find the overlap between gold and prediction sets. 
* Filler: Same as "Role"
* Binding: Since this is a (key-value) pair, the valid intersection would satisfy that both Role and Filler are intersected.

### Union
* Role: as what "Intersection" suggests, the entity is regarded as a set in both gold and predition, therefore, the union is all unique members of both gold and prediction sets. 
* Filler: Same as "Role"
* Binding: the union of binding is simply all (key, value) pairs of both gold and prediction sets.

### IOU
$$IOU = \frac{\text{Intersection}}{\text{Union}}$$

## Dataset
>[!NOTE] 
In CLAMS, all evaluations are regulated to utilize a specific batch of data. 

The evaluation process utilizes two batches of AAPB dataset:
* [BATCH #44](https://github.com/clamsproject/aapb-annotations/tree/89-rfb-gold/role-filler-binding/golds)
* [BATCH #90](https://github.com/clamsproject/aapb-annotations/blob/4f2eeb2ed838005528ee2a57ec3116cc544d7f12/batches/aapb-annotations-90.txt)

>[!IMPORTANT]
As of 8/16/2024, the evaluation is done to BATCH #44. Whereas, the batch is the training data for fine-tune the BERT model used in RFB. Thus, by doing such evaluation, we aimed to achieve to confirm two things: 1) the `evaluate.py` functions correctly. 2) the evaluation results should look decent.

## Results
The results of evaluation are stored in the file `results.txt` automatically after running `evaluate.py`. 

### Format
The unit of results is each video whose name is a unique GUID, and each video contains many frames which contain Role(s), Filler(s) to be assessed. Therefore, the format is structured as
```
GUID:
    <Frame_Num>: Role=<score> Filler=<score> Binding=<score>
```

### Sample result
```
cpb-aacip-398-57np5q9c:
		-1: Role=-1	Filler=-1	Binding=-1
cpb-aacip-41-10wpzpt5:
		52957: Role=1.0	Filler=0.0	Binding=0.0
		53167: Role=1.0	Filler=1.0	Binding=1.0
		53557: Role=1.0	Filler=0.0	Binding=0.0
		53647: Role=0.0	Filler=0.0	Binding=0.0
		53737: Role=0.5	Filler=0.0	Binding=0.0
cpb-aacip-283-61rfjj43:
		102738: Role=0.0	Filler=1.0	Binding=0.0
		102828: Role=0.5	Filler=0.33	Binding=0.33
		102917: Role=0.25	Filler=0.0	Binding=0.0
		103277: Role=0.67	Filler=0.67	Binding=0.67
		104596: Role=1.0	Filler=1.0	Binding=1.0
```
>[!NOTE]
Sometimes there's no common frames between gold and prediction in a single video, so we assign -1 to all entries and scores that should have been nonzero. For example, the first video in the sample result.

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

>[!WARNING] 
    As of 8/15/2024, the parallel mode hasn't been tested. So make sure include `in_seq` flag while running.