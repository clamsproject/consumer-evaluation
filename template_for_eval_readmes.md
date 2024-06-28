# Template for Future Evaluation Task Readmes
Description of task. 
The goal of this template is to increase the ease of running evaluations by future contributors and improving automation.  
Optional: Description on needed files.

## Required Input
* Date/Time in ISO Format
* [App/Model] (github commit link), version info.
* [Ground Truth/Golds Dataset] (github commit link/or describe where to get).
* [Prediction Dataset] (github commit link/or describe where to get)).
* [Evaluation Script] (github commit link).
* `Run command or instructions` that generated the report.

Describe the needed file types.  Recommended to use same format as report (above). 
How they are inputted - path to local dir, url to a github commit, one by one or by batch? (should be batch)  
Where are the needed files - which github dir for golds? (some are not publically accessible). 

> [!Note]: don't forget to set up a python environment for running the evaluation code.

## Usage
In the future, to run the evaluation code as a module, run the following command while in the `task_dir_name` directory:  
```bash
python3 -m evaluate -g <gold_url> -p <mmif_preds_dir> -r <result_file.txt>
# See `python3 evaluate --help` to check if there have been changes to the run command. 

```
Currently, the run commands may differ, and are generally of this format: 
```
# Example 1
python evaluate.py required-params-args 
```

## Output Format
Format of result if needed. 
Explanation of metrics can be in the `report.txt` or here. 
Since different evaluation runs may have different metrics, its suggested to place this in the report. 

## Important Notes / Common Issues / History
* Document any special changes that may be needed to run the code
* Document any common errors/issues if they are not already error-handled. 
* Document any notable history if needed.
