# For Developers of new Clams Apps Evaluations
Hello, it seems you are trying to create new evaluation code for a CLAMS app. 
To do so: 
```bash
# make the current /aapb_eval_module dir your current working directory. 
cd aapb_eval_module
# run the project directory generator
# the code will ask you to name the directory. Name it something ending with "_eval". e.g. "your_new_app_eval".
python generate_new_eval_dir.py
# download any requirements
cd .. 
cd your_new_app_eval
# edit the requirements.txt for any new requirements you would need.
pip install -r requirements.txt
```
This will create a new subdirectory for your evaluation task. It will contain: 
1. `__init__.py` for modular running of the app. 
2. `requirements.txt` for quickly downloading any requirements and common imports for writing eval code. 
3. `eval_abc.py` this is a copy of an abstract base class (abc) for you to begin writing the eval code with expected
required methods. 
4. `template_for_new_reports.md` as a pre-generated report template, again suggesting commonly expected parts of the 
report format for human reading. 
5. `template_for_new_readmes.md` as a pre-generated readme template, suggesting common part for task code clarification.








# Eval Module General Readme (WIP) 
The eval module is the concept to develop developer guidelines and helper code to write consistently similar evaluations
so that future evaluation workflows can be called automatically and easily.  

At the moment, this includes modularizing the whole aapb-eval repo, along with many of its subdirs which reflect different
eval tasks. However, future discussion could likely change how this calling should work. 
One discussed change is to have the run command take in a parameter for which/where-the subdir/task code to use to evaluate. 

## Contents
### In ./aapb_eval_module:  
 - `__init__.py` - for modularization.
 - `begin_new_eval_readme.md` - START HERE, starting instructions for new developers of new task eval code. 
 - `eval_abc.py` - abstract base class template which can be used to template new task eval code. 
 - `generate_new_eval_dir.py` - THEN HERE, RUN THIS. This is where one should start. It will create a new subdir for the new 
eval task, details are above. It will create a new code template for you to work in. It will create some of the other expected readmes, results.txt, etc. 
 - `goldretriever.py` - copy of the goldretriever, this is imported from this package in the provided code template. 
 - `locate_input.py` - new helper code in this package that should cover the common parts of the eval task. 
Unfortunately, most of the significant portions of writing the eval code are not common, as different file extractions and 
metrics are required for each new task. 
   - [!Note] - Getting the GUID_matches() function is likely common functionality and is left as a TODO. 

### In ./WIP:  
This was a sample eval to walk through the process of figuring out which steps were common and which were not. 
Recommended to delete this. 

### in the main directory (IMPORTANT): 
Because the PR for [readmes and templates](https://github.com/clamsproject/aapb-evaluations/pull/39) was not yet pulled into main 
the templates and readmes that are expected to be instructions and templates to be copied in `generate_new_eval_dir.py` 
are substituted by `./template_for_new_readmes.md` and `./template_for_new_reports.md`. These two should be deleted and replaced
with the prepared instructions and templates from the PR#39 above. 