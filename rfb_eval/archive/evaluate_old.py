#!/usr/bin/env python3

# Role-Filler Binding
# In-House Evaluation | CLAMS Team 2024

# ---------------------------------------------------------------------------|
# Imports
# ---------------------------------------------------------------------------|
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import csv
from io import StringIO
import json
import os 
import pandas as pd
import pathlib 

# local
#import goldretriever
import mmif

# type
from typing import Dict, Union, Tuple, Set, List
from mmif import Mmif


# ---------------------------------------------------------------------------|
# Data Loading
# ---------------------------------------------------------------------------|
def load_golds(gold_dir: Union[str, os.PathLike]) -> Dict:
    """Load Gold Documents

    Gold batches for RFB are expected to be a directory
    of JSON annotations

    ### params
    + gold_dir := directory containing gold annotation JSON files
    ### returns
    + a dictionary of guid-level annotations
    """
    reference_dir = pathlib.Path(gold_dir)
    golds = {}
    for ref_src in reference_dir.glob("*.?sv"):
        guid = pathlib.Path(ref_src).stem
        golds[guid] = process_gold(ref_src)
    return golds


def process_gold(fname: Union[str, os.PathLike]) -> Dict:
    """Process a single Gold document

    A single JSON represents a single video's 
    annotations - thus, a single gold JSON
    will map to a single MMIF pred file

    ### params 

    ### returns
    """
    gold_annotations = {}
    with open(fname, 'r', encoding='utf8') as f:
        json_obj = json.load(f)
        for frame, annotations in json_obj.items():
            gold_annotations[frame] = {k:v for k, v in annotations.items() if k[0] != "_"}
            #TODO Dean Cahill 06/24 - better way to handle skips than discarding?
    return gold_annotations


def load_preds(pred_dir: Union[str, os.PathLike]) -> Dict:
    """Load Pred Documents

    Pred batches for RFB are expected to be a directory
    of MMIF annotations, with a single MMIF representing
    a single video

    ### params
    + pred_dir := directory containing pred annotation MMIF files
    ### returns
    + a dictionary of frame-level RF pairs
    """
    pred_dir = pathlib.Path(pred_dir)
    preds = {}
    for pred_src in pred_dir.glob("*.mmif"):
        guid = pathlib.Path(pred_dir).stem 
        preds[guid] = process_pred(pred_src)
    return preds


def process_pred(
    pred_fname: Union[str, os.PathLike]
) -> Dict:
    """Convert MMIF

    Within MMIF, RFB output is a raw csv string.
    This function provides a simple means of collecting
    these strings into python dictionaries for comparison with golds

    ### params
    + mmif    := input mmif object to parse
    ### returns
    + a dictionary mapping frame numbers to RF pairs
    """
    vid_annotations = {}
    with open(pred_fname, 'r', encoding='utf8') as f:
        guid = pathlib.Path(pred_fname).stem
        frames = json.load(f)
        for framenum, annotations in frames.items():
            # TODO -> get csv string
            document_text = get_rfb_annotation(annotations)
            vid_annotations[guid] = convert_string_to_dict(document_text)

def get_rfb_annotation(annotations: Union[Mmif, Dict]):
    """Helper function to pull RFB-specific annotations
    out of a MMIF document.
    """
    if not isinstance(annotations, Mmif):
        mmif_obj = Mmif(annotations)

    # TODO - get RFB View
    # TODO - get TextDocument properties
    # TODO - get text @value property from textdoc


def convert_string_to_dict(csvstring: str) -> Dict[str, List[str]]:
    """Given a raw CSV string representing RF pairs,
    convert it to a dictionary of RF pairs
    """
    rf_pairs = defaultdict(list)
    df = pd.read_csv(StringIO(csvstring))
    
    for role, filler in zip(df["Role"], df["Filler"]):
        rf_pairs[role].append(filler)

    return rf_pairs

# ---------------------------------------------------------------------------|
# Evaluation
# ---------------------------------------------------------------------------|
class RFBMetrics:
    """
    A collection of metric functions for the Role-Filler Binding
    OCR structure parsing tool.

    ## Available Metrics

    + Dice-Sorensen Coefficient := F1-esque similarity
    + Intersection Over Union   := a simple in-house similarity metric
    """

    @staticmethod
    def dice_sorensen_coefficient(
        pred: Dict,
        gold: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """Dice Sørensen Coefficient

        ### params
        + pred := a single pred RF-pair sample (a frame's preds)
        + gold := a single gold RF-pair sample (a frame's golds)
        ### returns
        + average DSC for the frame
        + map of individual DSC values for each role in the frame
        """
        frame_level_dsc = 0.0
        individual_dsc_vals = {}

        for role, fillers in pred.items():
            if gold_fillers := gold[role]:
                p = set(fillers)
                g = set(gold_fillers)
                dsc = _dsc(p, g)
                frame_level_dsc += dsc
                individual_dsc_vals.update({role: dsc})

        def _dsc(pred: Set, gold: Set) -> float:
            num = 2 * len(set.intersection(pred, gold))
            denom = len(gold) + len(pred)
            try:
                dsc = num / denom
            except ZeroDivisionError as e:
                dsc = 0
            finally:
                return dsc

        frame_level_dsc /= len(gold)

        return frame_level_dsc, individual_dsc_vals

    @staticmethod
    def intersection_over_union(
        pred: Dict,
        gold: Dict,
        fuzzy: float = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Intersection Over Union

        ### params
        + pred := a single pred RF-pair sample (a frame's preds)
        + gold := a single gold RF-pair sample (a frame's golds)
        ### returns
        + average IOU for the frame
        + map of individual IOU values for each role in the frame
        """
        frame_iou = 0.0
        individual_iou_vals = {}

        # TODO => fuzzy matching
        for role, fillers in pred.items():
            if gold_fillers := gold[role]:
                predset = set(fillers)
                goldset = set(gold_fillers)
                iou = _iou(predset, goldset)
                frame_iou += iou
                individual_iou_vals.update({role: iou})

        def _iou(p, g):
            try:
                iou = len(set.intersection(p, g)) / len(set.union(p, g))
            except ZeroDivisionError:
                iou = 0
            finally:
                return iou

        return frame_iou, individual_iou_vals


# ---------------------------------------------------------------------------|
# Main
# ---------------------------------------------------------------------------|
def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Evaluation script for Role-Filler Binding",
    )

    parser.add_argument(
        "-p",
        "--preds",
        help="The directory location of prediction files",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--golds",
        help="The directory location of gold (JSON) files",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--metric",
        help="The metric being used for evaluation",
        required=True,
        default="dsc",
    )
    # TODO - add (more) runtime arguments

    return parser.parse_args()


def main(runtime_args: Namespace):
    # TODO run goldretriever + predretriever
    # TODO read in data
    # TODO primary evaluation function
    # TODO write results + report
    
    raise NotImplementedError


if __name__ == "__main__":
    test_string = ",Role,Filler\n0,Team,SCOTT VITBEUNBO\n1,Team,LALIE ONVOHOID\n2,Team,RICH HOYER\n3,Team,JOANNE LEGOMSKY\n"
    print(convert_string_to_dict(test_string))


    args = parse_args()
    main(args)
