import json
import os
import fnmatch
import goldretriever
from clams_utils.aapb import guidhandler
import csv
from brat_parser import get_entities_relations_attributes_groups  # install this dependency for tests
from mmif import Mmif
from typing import Any
from jiwer import wer, cer


# TODO: update ner to remove the source directory, it doesn't seem to be used
# TODO: implement a function for common evaluation metrics?
# TODO: merge these three classes
# TODO: remove the local goldretriever
# TODO: reconsider implementing something for side-by-side?


# STATIC EVALUATION TOOLS:
def filename_to_guid(filename) -> str:
    """Extracts the guid from a filename using the clams utils aapb extraction"""
    return guidhandler.get_aapb_guid_from(filename)


# CLASS-BASED EVALUATION TOOLS:
class OutputEval:
    def __init__(self, arguments, dict_data=None, str_data=None, process_pred=None, process_gold=None,
                 generate_confusion_matrix=None, gold_link=None):
        self.arguments = arguments
        self.dict_data = dict_data
        self.str_data = str_data

        self.test_dir = arguments.pred_file
        self.gold_dir = arguments.gold_file
        if self.gold_dir is None:
            self.gold_dir = self.get_gold(gold_link)
        self.test_files = os.listdir(self.test_dir)
        self.gold_files = os.listdir(self.gold_dir)

        # functions
        self.process_pred = process_pred  # function that generates pred data from pred file, like the text from a mmif
        self.process_gold = process_gold  # function that generates gold data from gold file
        self.confusion_matrix = generate_confusion_matrix  # function that generates confusion matrix for eval

    def write_data(self, dict_data):
        """Adds the evaluation data to the class"""
        self.dict_data = dict_data

    def write_results(self):
        """Write evaluation results to either a dict or string, to be outputted as a txt file"""
        try:
            with open(self.arguments.result_file, 'w') as fh_out:
                if self.dict_data:
                    json.dump(self.dict_data, fh_out, indent=4)
                    fh_out.write("\n")
                if self.str_data:
                    fh_out.write(self.str_data)
        except AttributeError as e:
            print(f"Error result_file not in args: {e}")

    def get_gold(self, gold_link):
        """If the gold file wasn't listed, get the gold file from the goldretriever"""
        try:
            if self.arguments.gold_file is None:
                self.arguments.gold_file = goldretriever.download_golds(gold_link)
        except AttributeError as e:
            print(f"Error either a directory or link to directory must be provided for gold files: {e}")
        return self.arguments.gold_file

    def match_files(self) -> list[tuple]:
        """Compare the files in the gold and test directories. Return pairs of matching files in a list.
        :param test_dir: Directory of test .mmif files
        :param gold_dir: Directory of gold .tsv files
        :return: list of tuples containing corresponding data file locations in (test, gold) format.
        """
        file_matches = []
        for gold_file in self.gold_files:
            pattern = gold_file[:24] + "*"
            for test_file in self.test_files:
                if fnmatch.fnmatch(test_file, pattern):
                    gold_match = os.path.join(self.gold_dir, gold_file)
                    test_match = os.path.join(self.test_dir, test_file)
                    file_matches.append((test_match, gold_match))
                    self.test_files.remove(test_file)
                    break

        return file_matches

    def load_references(self) -> dict[str, tuple[Any, Any]]:
        """Processes each gold and pred file at the same time
            :return: list of tuples containing corresponding processed data in (pred, gold) format, keyed by the guid
        """
        refs = {}

        for match in self.match_files():
            pred, gold = match

            # get the guid
            guid = filename_to_guid(pred)

            # process the preds
            mmif = Mmif(open(pred).read())
            pred_data = self.process_pred(mmif)

            # process the golds
            with open(gold, 'r', encoding='utf8') as f:
                # identify the file type
                if gold.endswith('.ann'):
                    entities, relations, attributes, groups = get_entities_relations_attributes_groups(gold)
                    reader = (entities, relations, attributes, groups)
                elif gold.endswith('.txt'):
                    reader = f.read()
                else:  # for csv or tsv
                    reader = csv.reader(f)

                # process the file
                gold_data = self.process_gold(reader)

            refs[guid] = (pred_data, gold_data)

        return refs

    def run_eval(self, eval_methods: tuple, confusion_matrix=None):
        """Calculates the evaluations
            :param eval_methods: a tuple of the function(s) for the type of eval we are doing in the form of strings
            :param confusion_matrix: confusion matrix for evaluation metrics (tp, fp, fn, tn)
            :return output_dict: a dictionary of the evaluated output for each guid
        """
        data = self.load_references()
        result = []

        # calculating the results
        for guid in data:
            print("Processing file: ", guid)

            try:
                for eval_method in eval_methods:
                    if eval_method == "wer" or eval_methods == "cer":
                        if eval_method == "wer":
                            eval_method = self.calc_wer
                        if eval_method == "cer":
                            eval_method = self.cer_by_timeframe
                        result_exact_case = eval_method(data[guid][0], data[guid][1], True)
                        result_ignore_case = eval_method(data[guid][0], data[guid][1], False)
                        result.append((guid, result_exact_case, result_ignore_case))
            except Exception as exception:
                print("Error processing file: ", guid, exception)

        # generating the output
        output_dict = {}

        # if we have document-level eval
        werS_sum = 0
        werI_sum = 0
        for r in result:
            output_dict[r[0]] = {'WER-case-sensitive': r[1], 'WER-case-insens': r[2]}
            werS_sum += r[1]
            werI_sum += r[2]
        if len(result):
            output_dict['Average'] = {'WER-case-sensitive': werS_sum / len(result),
                                      'WER-case-insens': werI_sum / len(result)}

        # if we have basic metrics
        if "basic_metrics" in eval_methods:
            outputs = self.basic_metrics()
            output_dict["Precision"] = outputs[0]
            output_dict["Recall"] = outputs[1]
            output_dict["F1"] = outputs[2]

        return output_dict

    def casing_text(self, text, ignore_case):
        """Includes any text preprocessing we would like to add, currently only casing"""
        if ignore_case:
            text = text.upper()
        return text

    def calc_wer(self, pred_text, gold_text, exact_case):
        """Document-level wer calculation
            :param exact_case: whether we want to consider casing for wer calculation
            :param pred_text: predicted text
            :param gold_text: gold text
            :return wer: document-level wer score
        """
        pred = self.casing_text(pred_text, not exact_case)
        gold = self.casing_text(gold_text, not exact_case)
        return wer(pred, gold)

    def cer_by_timeframe(self, pred: dict[float, str], gold: dict[tuple, str], exact_case: bool):
        """Document-level cer calculation
            :param pred: text for each timespan in a dict
            :param gold: text for each timespan in a dict
            :param exact_case: whether we want to consider casing for wer calculation
            :return vals: dict of each timespan, pred and gold texts, and their cer score
        """
        vals = {}
        for timepoint, text in pred.items():
            for timespan, gold_text in gold.items():
                start = float(timespan[0])
                end = float(timespan[1])
                if start <= timepoint < end:
                    if start not in vals:
                        vals[start] = {'ref_text': gold_text, 'hyp_text': text}
                    else:
                        if len(vals[start]['hyp_text']) < len(text):
                            vals[start]['hyp_text'] = text
            for comp in vals.values():
                pred_text = self.casing_text(comp['hyp_text'], not exact_case)
                gold_text = self.casing_text(comp['ref_text'], not exact_case)
                comp['cer'] = str(cer(pred_text, gold_text))
        return vals

    # This is a naive implementation I got from sr eval, should we consider lightweight packages like huggingface's
    # evaluation libraries?
    def basic_metrics(self):
        tp = self.confusion_matrix[0]
        fp = self.confusion_matrix[1]
        fn = self.confusion_matrix[2]

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        return precision, recall, f1

