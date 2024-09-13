# AAPB Evaluations 
This repository contains the evaluation codebase, results, and reports for the AAPB-CLAMS collaboration project. Evaluations are done on [CLAMS Apps](apps.clams.ai/) or on a pipeline/group of CLAMS Apps that give an evaluable result on a certain task for video metadata extraction.  

## Structure of This Repository
Each subdirectory of the repository is an evaluation task within the project. Each has its own files as above. 

### Filename Conventions 
#### Inputs to Evaluations
* golds - The gold standard, human-annotated files by which apps are evaluated for predictive ability. 
  * synomymous "ref", "reference", "groundtruth", or "goldstandard"
  * often are `.tsv` or `.csv` or `.txt`. 
* predictions - The app-predicted files with predicted-annotations of what phenomena are to be evaluated. (e.g. time durations for slate detection.)
  * synomymous "pred", "test", "system", or "output"
  * each preds directory represents a batch, with naming conventions as follows:`preds@<APP_NAME><APP_VER>@<BATCH_NAME>`
  * are always `.mmif` files with app views. 
#### Outputs to Evaluations
* results - This should be the result system output of the evaluation numbers from a finished evaluation. 
  * often `results.txt` file. This should be renamed according to conventions currently listed [here](/template_for_eval_reports.md).
  * This was used before to describe machine out prediction "results". THIS TERM NO LONGER REFERS TO THIS.  
  * There might be results per GUID, or it may be a summary. 
* reports - Reports are more formal documents that describe the results meant for business intelligence.
  * Plans are to automate some generation of the report from the results, which may require some automatic scripts. However, some parts of the report must often be manually curated. 

_See Remaining Work for continued filename convention issues._

## Workflow of Evaluations
> [!Important]
> In the future, many of the evaluations should also retrieve the golds automatically by using `from clams_utils.aapb import goldretriever` and `goldretriever.download_golds(<params>)`. Thus, it is usually not required to provide -g. 

1. Choose evaluation task, create batch with GUIDs.
2. In [AAPB Annotations](https://github.com/clamsproject/aapb-annotations), create raw annotations, then `process.py` into golds. Upload those golds via a github commit. (Requires preprocessing and access to videos)
3. Run app/pipeline-of-apps to create output pred `.mmif`s locally on your machine. (Also requires access to videos)
4. Run `evaluator.py` with the appropriate evaluation type `-e` and inputting ` -m path-to-local-mmifs -g url-to-golds-commit`, and any other necessary arguments. Obtain result files. 
5. `evaluator.py` will generate summary of results in the form of `generate summary of results` or a directory, depending on the evaluation type.

### Instructions to Run Apps
[CLAMS Apps Manual](https://apps.clams.ai/clamsapp/).  
[TestDrive Instructions (Alternate)](https://gist.github.com/keighrim/5e97a41a40d623d6ad4f1d0e325786a9).

### Instructions to run `evaluator.py`
Command structure:
```
python evaluator.py -e <type_eval> -m <path-to-preds> -g <path-to-golds> additional arguments...
```
Example command:
```
python evaluator.py -e timeframe_eval -m timeframe_eval/test-slate-preds -g golds/timeframe-slate-test --slate
```

This script takes the arguments necessary for each evaluation type. To run this evaluation script, you need the following:
* `-e`: Type of evaluation to run, see supported evaluation packages and their additional necessary arguments below. (i.e. 'asr_eval', currently will not accept 'asr')
* `-m`: Set of predictions in MMIF format (either from the preds folder in repo of the evaluation, or generated from one of the [CLAMS apps](https://apps.clams.ai)     )
* `-g`: Set of golds in csv format (either downloaded from the annotations repository using goldretriever.py, or your own set that exactly matches the format present in aapb-annotations)

There are currently 7 supported evaluation packages. The following is a list along with the arguments necessary to them (in order for implementing) in addition to `-m` and `-g`:
* Automatic Speech Recognition `asr`
* Forced Alignment `fa` 
  * `-r`, `--result_file`: file to store evaluation results
  * `-t`, `--thresholds`: comma-separated thresholds in seconds to count as "near-miss", use decimals
* Named-entity Linking `nel`
  * `-r`, `--result_file`: file to store evaluation results 
* Named-entity Recognition `ner`
  * `--side-by-side`: directory to publish the side by side comparison  
  * `-r`, `--result_file`: file to store evaluation results 
  * `--source_directory`: directory that contains original source files (without annotations)
* Optical Character Recognition `ocr`
  * `-r`, `--result_file`: file to store evaluation results  
* Scene Recognition `sr`
  * `--count_subtypes`:  bool flag whether to consider subtypes for evaluation
* Timeframes `timeframe`
  * `--slate` / `--chyron`: use one of these flags so that timeframe either evaluates for slate or chyron annotations
  * `--side-by-side`: directory to publish the side by side comparison
  * `-r`, `--result_file`: file to store evaluation results

If there is every confusion on which arguments to use for a given evaluation, call the `--specific-help` flag after naming the evaluation type `-e` to see which arguments it uses, as shown in the command below:
```
python evaluator.py -e ner_eval/ --specific-help
```

The output of these evaluations currently depends on the evaluation type, see the relevant evaluation package for it's specific output type.

## Remaining Work
The users and use cases of this evaluation workflow remain under discussion. For the moment, the work expected has been converted into [issues](https://github.com/clamsproject/aapb-evaluations/issues).  
