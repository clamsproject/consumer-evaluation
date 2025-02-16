import argparse

import typing
import os
import sys
# Get the current file's directory (where this script is located)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one directory to get the root directory
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(root_dir, '..'))
# Add the root directory to sys.path
sys.path.append(root_dir)
# from aapb_eval_module import goldretriever
import aapb_eval_module

# import pandas as pd

# from clams_utils.aapb import goldretriever
# source ../clams_env/bin/activate







if __name__ == "__main__":
    # EVALUATION TASK/MODEL INFORMATION (Edit these to needed values.) ====
    APPNAME = "parseqocr-wrapper"  # default app
    APPVERSION = 1.0  # default version
    GOLDS_DIR = 'https://github.com/clamsproject/aapb-annotations/tree/f884e10d0b9d4b1d68e294d83c6e838528d2c249/newshour-transcript-sync/golds/aapb-collaboration-21'

    # Parse CLI ====
    parser = argparse.ArgumentParser(description="This sample evaluation of <task> is written to test modularization of the evals utils.\n"
                                                 "An example run is\n"
                                                 "'py -m sample_eval -g /google.com -p eval_module/ -r resultssss.txt'")
    parser.add_argument("-g", "--golds-dir", help="github commit url link to gold standard <filetype> data dir, if None will goldretrieve from aapb-annotations for evaluation standard.", default=None)
    parser.add_argument("-p", "--preds-dir", help="path to local dir with machine-prediction <filetype> files to be evaluated.")
    parser.add_argument("-r", "--res", help="filepath/name for output results from this evaluation", default="results.txt") #could add datetime of current run.
    # parser.add_argument("-o", "--out-dir", help="path for a results dir", default=None) # add this line if an output dir is needed.
    args = parser.parse_args(); # print(args)

    golds_dir, preds_dir, out_dir = aapb_eval_module.locate_input.locate(args, APPNAME, APPVERSION, GOLDS_DIR)
    # outdir = None. #results directory?? created here??

    # get individual entries from golds and preds =====
    # get golds files
    # get preds files
    # collect out entries
    # pair





# ======
# TODO:
'''
Progress: Got module and paths to work. 
Created dummy reads. Now fix if the Path globs are correct. 
Consider which eval I want to emulate. 
Consider if we can move this get code to the eval_module. 
'''

'''
# Importing modules from the package
from my_package import module1
from my_package.subpackage import sub_module

# Using the imported modules
module1.some_function()
sub_module.another_function()
'''