import json
import os
import fnmatch
import goldretriever
from clams_utils.aapb import guidhandler
import csv
from brat_parser import get_entities_relations_attributes_groups  # install this dependency for tests
from mmif import Mmif
from typing import Any
from pyannote.metrics.diarization import DiarizationCoverage, DiarizationPurity
from pyannote.metrics.segmentation import SegmentationCoverage, SegmentationRecall, SegmentationPrecision, \
    SegmentationPurity


# TODO: implement a function for common evaluation metrics?
# TODO: merge these three classes
# TODO: see if we can standardize the use of goldretriever and the other stuff in the main() of eval scripts
# TODO: reconsider implementing something for side-by-side?

# going to start working on standardizing metrics next
# then will combine all of these annotations


# STATIC EVALUATION TOOLS:
def filename_to_guid(filename) -> str:
    """Extracts the guid from a filename using the clams utils aapb extraction"""
    return guidhandler.get_aapb_guid_from(filename)


# CLASS-BASED EVALUATION TOOLS:
class OutputEval:
    def __init__(self, arguments, dict_data=None, str_data=None):
        self.arguments = arguments
        self.dict_data = dict_data
        self.str_data = str_data

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
        if self.arguments.gold_file is None:
            self.arguments.gold_file = goldretriever.download_golds(gold_link)
        return self.arguments.gold_file


class PreProcessEval:
    def __init__(self, test_dir, gold_dir):
        self.test_dir = test_dir
        self.gold_dir = gold_dir
        self.test_files = os.listdir(self.test_dir)
        self.gold_files = os.listdir(self.gold_dir)

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

    def load_references(self, process_pred, process_gold) -> dict[str, tuple[Any, Any]]:
        """Processes each gold and pred file at the same time
            :param process_gold: function that generates gold data from gold file
            :param process_gold: function that generates pred data from pred file
            :return: list of tuples containing corresponding data file locations in (test, gold) format, keyed by the guid
        """
        refs = {}

        for match in self.match_files():
            pred, gold = match

            # get the guid
            guid = filename_to_guid(pred)

            # process the preds
            mmif = Mmif(open(pred).read())
            pred_data = process_pred(mmif)

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
                gold_data = process_gold(reader)

            refs[guid] = (pred_data, gold_data)

        return refs


class MetricsEval:
    def __init__(self):
        ...

    # from ASR
    def wer(self):
        ...

    # from ocr
    def cet(self):
        ...

    # from nel, sr, timeframe eval
    def basic_metrics(self):
        # precision
        # recall
        # f1

        # include micro and macro average?
        ...

    # from FA
    def timeframe_detection_metrics(self):
        coverage = DiarizationCoverage()
        purity = DiarizationPurity()
        scoverage = SegmentationCoverage()
        spurity = SegmentationPurity()
        precision = SegmentationPrecision()
        recall = SegmentationRecall()
        ...