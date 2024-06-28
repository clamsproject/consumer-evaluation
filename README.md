# AAPB Evaluations 
This repository contains the evaluation codebase, results, and reports for the AAPB-CLAMS collaboration project. Evaluations are done on [CLAMS Apps](apps.clams.ai/) or on a pipeline/group of CLAMS Apps that give an evaluable result on a certain task for video metadata extraction.  

## Structure of This Repository
Each subdirectory of the repository is an evaluation task within the project. Each has its own files as above. 

### Filename Conventions 
#### Inputs to Evaluations
* golds - The gold-truth standard, humanly-annotated files by which apps are evaluated for predictive ability. 
  * often are `.tsv` or `.csv` or `.txt`. 
* preds/predictions - The app-predicted files with predicted-annotations of what phenomena are to be evaluated. (e.g. time durations for slate detection.)
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
> In the future, evaluations should be invoked in the same manner. Likely through a Docker module, or via a CLI command that is the same.  
> `cd into the appropriate task_eval directory`
> example command: `python3 -m evaluate -g url/to/gold/web -p path/to/local/mmifpreds -r results_printout_filename.txt` 
> Many of the evaluations should also retrieve the golds automatically by using `from clams_utils.aapb import goldretriever` and `goldretriever.download_golds(<params>)`. Thus, it is usually not required to provide -g. 

1. Choose evaluation task, create batch with GUIDs.
2. In [AAPB Annotations](https://github.com/clamsproject/aapb-annotations), create raw annotations, then `process.py` into golds. Upload those golds via a github commit. (Requires preprocessing and access to videos)
3. Run app/pipeline-of-apps to create output pred `.mmif`s locally on your machine. (Also requires access to videos)
4. Run the evaluation code inputting **url-to-golds-commit** and **path-to-local-mmifs**. Obtain result files. 
5. Have python code and hand-generation of `nameconvention-report.txt` to generate summary of results. 

### Instructions to Run Apps
[CLAMS Apps Manual](https://apps.clams.ai/clamsapp/).  
[TestDrive Instructions (Alternate)](https://gist.github.com/keighrim/5e97a41a40d623d6ad4f1d0e325786a9).

## Remaining Work
The users and use cases of this evaluation workflow remain under discussion. For the moment, the work expected has been converted into [issues](https://github.com/clamsproject/aapb-evaluations/issues).  
