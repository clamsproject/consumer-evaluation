'''Abstract Base Class for new Evaluations of Clams Apps
There are two helper code modules for writing new clams app evaluations.
1. This abstract base class and inherited class for writing the new evaluation code. This has methods that are
suggestively required steps for a new evaluation code that are not standardized.
2. Some standardized/common functions have been implemented for you and are contained in this eval module as a package.
Furthermore, the other pieces are imported here. These include things like: goldretriever, locate_inputs.
Moving forward, common accuracy metrics from known libraries will be included in the import as well.

To use this template, run "generate_new_eval.py <project_name> to create a project directory. Add any extra dependencies
you want in the requirements.txt.  Then run "pip install -r requirements.txt" to import all the suggested modules/code
along with any new requirements you require.

'''

from abc import ABC, abstractmethod
import aapb_eval_module # import its own module when copied to a new dir.
import argparse
# add general imports here.

class NewEvaluation(ABC):
    @abstractmethod
    def define_runtime_cli_args(self):
        pass

    @abstractmethod
    def get_entries_from_gold(self):
        pass

    @abstractmethod
    def get_entries_from_preds(self):
        pass

    @abstractmethod
    def compare_entries(self):
        pass

    @abstractmethod
    def write_results(self):
        pass

    @abstractmethod
    def write_to_report(self):
        pass

class MyEvaluation(NewEvaluation): # change the name of MyEvaluation
    def define_runtime_cli_args(self):
        print("Define the arguments that will be called during the running of your specific eval."
              "The expected arguments are -gpr for gold, preds, results for the locations of two inputs and one."
              "Call locate_input.locate() with -g and -p strings passed."
              "Locate() will gather a list of guids that match, ones that dont (you can do error handling of this),"
              "and lists of the gold files and mmif files with their locations, so you can begin to extract"
              "individual entries of phenomena from them to compare/evaluate. ")
        # guid_list, guid_mismatch, gold_files, mmif_files = locate_input.locate(args)

    def get_entries_from_gold(self):
        print("Given gold file list,"
              "Return list of the individual items-to-be-evald from gold files")

    def get_entries_from_preds(self):
        print("Give preds file list,"
              "Return list of the individual items-to-be-evald from gold files")

    def compare_entries(self):
        print("Match and Compare the entries together")

    def write_results(self):
        print("Gather results from compare_entries, write them to a output directory, "
              "suggested one result file per guid.")

    def write_to_report(self):
        print("Gather results from compare_entries, combine them into aggreggate results and write long form and"
              "short form to a report file.")

# You cannot create an instance of an abstract class
# obj = MyAbstractClass()  # This would raise an error

# Instantiate the concrete class
obj = MyEvaluation() # change the name of MyEvaluation

# Call the implemented methods
obj.define_runtime_cli_args()
obj.get_entries_from_gold()
# etc


'''
Parts of Eval: 
enter argparse arguments during run (this should be templated!! - lets use comments in the abc?)
 - Differentiate if they are paths or urls
create outdir if needed? (templated!!)
call locate_input.locate() with 3 args: -gpr # should likely be only 2. (comment in abc)
 - locate calls goldretriever()
 - locate calls predretriever() # create this
 - locate calls get_guid_matches() which tests if similar. 

individual eval code - input guid_matches, gold_files, mmif_files
 - gets entries from golds (abst method)
 - gets entries from mmifs (abst method)
 - compare entries (abst method)
 - get results, metrics 
 - writes to outdir/results (abst method)
 - print some results to report (abst method) # future: possibly take some metrics and directly print to report. 
 

 




'''
