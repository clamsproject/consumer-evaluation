"""Named Entity Linking Evaluation

$ evaluate.py [-h] [-o [OUTPUT]] [test-dir] [gold-dir]

Evaluate .mmif files with named entity linking annotations produced as output
from the DBpedia Spotlight wrapper app. Compares system
generated data with the gold data found in the aapb-annotations
repository.

NOTE: gold annotation files and test output files that correspond
to the same AAPB catalog item must begin with the same GUID.
i.e. `gold-files/cpb-aacip-507-1v5bc3tf81.tsv` and `test-files/cpb-aacip-507-1v5bc3tf81-transcript.txt.dbps.mmif

"""

import argparse
from collections import defaultdict
import fnmatch
import json
from lapps.discriminators import Uri
from mmif.serialize import Mmif
import os
import pandas as pd

import goldretriever
from nel import NamedEntityLink

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils import standardized_parser, basic_eval


def match_files(test_dir, gold_dir) -> list:
    """Compare the files in the gold and test directories. Return pairs of matching files in a list.
    :param test_dir: Directory of test .mmif files
    :param gold_dir: Directory of gold .tsv files
    :return: list of tuples containing corresponding data file locations in (test, gold) format.
    """
    test_files, gold_files = os.listdir(test_dir), os.listdir(gold_dir)
    file_matches = []
    for gold_file in gold_files:
        pattern = gold_file[:24] + "*"
        for test_file in test_files:
            if fnmatch.fnmatch(test_file, pattern):
                gold_match = os.path.join(gold_dir, gold_file)
                test_match = os.path.join(test_dir, test_file)
                file_matches.append((test_match, gold_match))
                test_files.remove(test_file)
                break

    return file_matches


def filter_nil_entities(gold_tsv) -> list:
    """Returns list of gold entities excluding the ones without grounding."""
    gold_entities = file_to_ne(gold_tsv)
    # remove gold NEL instances whose QIDs are empty strings
    gold_entities = [gold_ent for gold_ent in gold_entities if gold_ent.kbid]

    return gold_entities


def file_to_ne(file_path: str) -> list:
    """Checks whether the file is in .mmif or .tsv format and calls the appropriate function
    to get a list of NEL objects"""
    if file_path.endswith('.mmif'):
        return mmif_to_ne(file_path)
    elif file_path.endswith('.tsv'):
        return tsv_to_ne(file_path)
    else:
        raise Exception("Unsupported file type.")


def mmif_to_ne(mmif_path) -> list:
    """Fetch named entities from the input mmif.
    Returns a list of NEL objects.
    """
    with open(mmif_path) as fh_in:
        mmif_serialized = fh_in.read()
    mmif = Mmif(mmif_serialized)
    ne_views = mmif.get_all_views_contain(Uri.NE)
    view = ne_views[-1]  # read only the first view (from last) with Uri.NE
    annotations = view.get_annotations(at_type=Uri.NE)
    ne_list = []
    guid = os.path.basename(mmif_path)[:24]
    for anno in annotations:
        entity = anno.properties
        ne = NamedEntityLink(guid, entity['start'], entity['end'], entity['category'], entity['text'],
                             entity['grounding'][1:])
        ne_list.append(ne)

    return ne_list


def tsv_to_ne(gold_tsv_path) -> list:
    """Fetch named entities from the input tsv.
    Returns a list of NEL objects."""
    with open(gold_tsv_path) as fh_in:
        annotations_df = pd.read_csv(fh_in, sep='\t', encoding='utf-16')
        annotations_df.fillna('', inplace=True)
        annotations_df['guid'] = annotations_df['guid'].apply(lambda x: x[:24])
    ne_list = [NamedEntityLink(guid, begin_offset, end_offset, ent_type, text, uris) for
               guid, begin_offset, end_offset, ent_type, text, uris in
               zip(annotations_df['guid'], annotations_df['begin_offset'], annotations_df['end_offset'],
                   annotations_df['type'], annotations_df['text'], annotations_df['qid'])]

    return ne_list


def evaluate(test_dir, gold_dir=None):
    if gold_dir is None:
        gold_dir = goldretriever.download_golds('https://github.com/clamsproject/aapb-annotations/tree/feaf342477fc27e57dcdcbb74c067aba4a02e40d/newshour-namedentity-wikipedialink/golds/aapb-collaboration-21')
    results = defaultdict(dict)
    file_matches = match_files(test_dir, gold_dir)
    for sys_file, gold_file in file_matches:
        print(f'>>> Evaluating {os.path.basename(sys_file)}')
        guid = os.path.basename(sys_file)[:24]

        sys_instances = frozenset(file_to_ne(sys_file))
        gold_instances = frozenset(filter_nil_entities(gold_file))
        results[guid]['Gold Entities'] = {"count": len(gold_instances)}
        results[guid]['System Entities'] = {"count": len(sys_instances)}

        # calculate precision
        precision = len(gold_instances.intersection(sys_instances)) / len(sys_instances)

        # calculate recall
        recall = len(gold_instances.intersection(sys_instances)) / len(gold_instances)

        # calculate F1
        if precision + recall == 0:  # avoid ZeroDivisionError
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        results[guid]['Precision'] = "{:.2f}".format(precision)
        results[guid]['Recall'] = "{:.2f}".format(recall)
        results[guid]['F1'] = "{:.2f}".format(f1)
        print('>>> ... done')

    return results


def write_results(data: dict, result_path: str):
    """Write evaluation results to txt file."""
    with open(result_path, 'w') as fh_out:
        json.dump(data, fh_out, indent=4)


if __name__ == "__main__":
    args = standardized_parser.parse_args()
    data = evaluate(args.pred_file, args.gold_file)

    evaluator = basic_eval.Eval(args, dict_data=data)
    evaluator.write_results()
