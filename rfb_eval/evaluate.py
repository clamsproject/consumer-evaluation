"""Description
This script is to complete evaluation process for the app RFB(role-filler-binding).
Used evaluation metrics include:
    + Intersection Over Union (IOU)
"""

from abc import ABC, abstractmethod
import json
import os
from typing import Dict, Set, Optional, Tuple, Union, List
from collections import defaultdict
from io import StringIO
from argparse import ArgumentParser, Namespace
import logging
import multiprocessing as mp

from thefuzz import fuzz
from mmif import Mmif, AnnotationTypes, View, Annotation
from mmif.utils import video_document_helper as vdh
from clams_utils.aapb import guidhandler, goldretriever #TODO: goldretriver will be used after the URL for the gold-standard data is available

import pandas as pd
import numpy as np

GOLD_URL = 'https://github.com/clamsproject/aapb-annotations/tree/89-rfb-gold/role-filler-binding/golds'
SWT_APP = 'http://apps.clams.ai/swt-detection/v6.1'
RFB_APP = 'http://apps.clams.ai/role-filler-binder/v1.0'

#--------------------------------------------------------------------
# Functions needed to load predictions made by RFB
#--------------------------------------------------------------------
def get_adj_aligned_ann(ann: Annotation, view: View) -> Optional[Annotation]:
    """
    Get the aligned annotation residing in adjacent view

    :param ann: any type of annotation
    :param view: the view that contains both input annotation and alignments

    :return: either None type or an aligned annotation found in adjacent view
    """
    for al in view.get_annotations(AnnotationTypes.Alignment):
        if aligned_ann := ann.aligned_to_by(al):
            return aligned_ann

def get_aligned_ann_of(
    mmif: Mmif,
    source: Annotation,
    source_app: str,
    target_app: str
    ) -> Optional[Annotation]:
    """
    Get the aligned annotation of the input annotation cross views in MMIF

    :param: mmif: the mmif file
    :param: source: the source annotation that looks for its aligned annotation in target view
    :param: source_app: the app that contains source annotation
    :param: target_app: the app that might contain target aligned annotation

    :return: either None type or an annotation
    """
    valid_views = {view.metadata.app: view
                   for view in mmif.views if not (view.has_error() or view.has_warnings())}

    # Validate if two apps are in mmif
    if not (source_app and target_app) in valid_views:
        raise ValueError(f"Either {source_app} or {target_app} is not in mmif")
    #TODO: Think about more edge cases
    current_view, target_view = valid_views[source_app], valid_views[target_app]
    current_ann = source
    while current_view.id != target_view.id:
        next_ann = get_adj_aligned_ann(current_ann, current_view)
        current_view = mmif.get_view_by_id(next_ann.parent)
        current_ann = next_ann
    return current_ann

def csv_string_to_pair(csv_string: str) -> Set[Tuple[str, str]]:
    """
    Convert csv-string to a set of pairs represeted by tuples

    :param: csv_string: the csv-formatted string
    :return: a set of tuples of (role, filler) string
    """
    return set(pd.read_csv(StringIO(csv_string), index_col=0).fillna('nan').itertuples(index=False, name=None))  #FIXME: Empty role/filler is filled with string 'nan'

def load_pred(file: Union[str, os.PathLike]) -> Dict[str, Dict]:
    """
    Load the predicted role-filler pair made by RFB

    :param: file: the file path or name of the RFB MMIF
    :return: a nested dictionary data structure that indexes GUID -> frame_num -> (role, filler)
    """
    guid = guidhandler.get_aapb_guid_from(file)
    logging.debug("Loading prediction data for %s...", guid)

    with open(file, encoding='utf-8') as f:
        rfb_mmif = Mmif(json.load(f))
    f.close()
    rfb_view = rfb_mmif.views.get_last_contentful_view()

    frames_dict = {}
    for rfb_td in rfb_view.get_documents():
        aligned_tp = get_aligned_ann_of(rfb_mmif, rfb_td, RFB_APP, SWT_APP)
        aligned_frame = vdh.convert_timepoint(rfb_mmif, aligned_tp, 'frames')
        frames_dict[aligned_frame] = csv_string_to_pair(rfb_td.text_value)

    return {guid: frames_dict}

#--------------------------------------------------------------------
# Functions needed to load gold standard data
#--------------------------------------------------------------------
def csv_string_to_set(csv_string: str) -> Set[Tuple[str, str]]:
    """
    Convert csv-string to a set of tuples which represent (role, filler) pairs

    :params: csv_string: the input csv-formatted string
    :return: a set of tuples of (role, filler)
    """
    rf_set = set()
    for pair in csv_string.split('\n'):
        _, role, filler = pair.split(',', maxsplit=2)
        rf_set.add((role, filler))
    return rf_set

def load_gold(gold_csv: Union[str, os.PathLike]) -> Dict[str, Dict]:
    """
    Load gold-standard csv data for RFB

    As a review, its format looks like:
    GUID | FRAME | SWT-TYPE | SKIPPED | ANNOTATIONS
    -----------------------------------------------
    str  | int   |   str    |  T/F    | csv-string
    """
    guid = guidhandler.get_aapb_guid_from(gold_csv)
    logging.debug("Loading gold-standard data for %s...", guid)
    frames_dict = defaultdict(set)
    df = pd.read_csv(gold_csv).dropna(subset=['ANNOTATIONS'])  #FIXME: Empty annotations are dropped from this process

    min_frame, max_frame = -1, -1
    anns = set()
    for _, frame in df.iterrows():
        if not frame['SKIPPED']:
            if anns:
                frames_dict[(min_frame, max_frame)] = anns
            anns = csv_string_to_set(frame['ANNOTATIONS'])
            min_frame = frame['FRAME']
        else:
            if frame['ANNOTATIONS'] == 'DUPLICATE':
                max_frame = frame['FRAME']
            else:
                if anns:
                    frames_dict[(min_frame, max_frame)] = anns
                    anns = set()

    return {guid: frames_dict}

#--------------------------------------------------------------------
# Class of evaluation metrics
#--------------------------------------------------------------------
class RFBMetrics(ABC):
    """
    The abstract class of evaluation metrics for evaluating a single video run by RFB
    """
    def __init__(self, gold: Dict[str, Dict], pred: Dict[str, Dict]) -> None:
        self.gold = gold
        self.pred = pred
        self.frame_score: Dict[int, Tuple] = {}

        # Check if the gold and pred data refer to the same video
        if list(self.gold.keys())[0] != list(self.pred.keys())[0]:
            raise ValueError('The prediction file and the gold file refer to different video.')

        # Find intersected frames between pred and gold
        self.guid = next(iter(self.pred.keys()))
        self.frames = self._align_frames_between(self.pred, self.gold)

    def _align_frames_between(self,
                              pred: Dict[str, Dict[int, set]],
                              gold: Dict[str, Dict[Tuple[int, int], set]]
                              ) -> Dict[int, Tuple[int, int]]:
        """Help aligning frames from predictions with span from golds"""
        alignments = defaultdict(tuple)
        for frame in pred[self.guid]:
            for span in gold[self.guid]:
                if span[0] <= frame <= span[1]:
                    alignments[frame] = span
                    break

        return alignments

    @abstractmethod
    def calculate(self) -> float:
        pass

class StringList:
    """
    The class that represents a list of strings and it enables the "outer product" operation
    """
    def __init__(self, strs: List[str]) -> None:
        self.strs = strs

    def __matmul__(self, other: List[str]) -> np.ndarray:
        if not isinstance(other, StringList):
            raise ValueError("The right-hand operand must be an instance of StringList")

        # Perform the "outer product" operation and store results in a list of lists
        result_matrix = []
        for str1 in self.strs:
            row = []
            for str2 in other.strs:
                distance = fuzz.ratio(str1, str2)
                row.append(distance)
            result_matrix.append(row)

        # Convert the result matrix to a NumPy array
        return np.array(result_matrix)

class IOU(RFBMetrics):
    """
    The class of Intersection Over Union (IOU) evaluation metric
    """
    def _iou(self, p: set, g: set) -> float:
        try:
            val = len(set.intersection(p, g)) / len(set.union(p, g))
        except ZeroDivisionError:
            val = 0
        return val

    def _organize(self, frame_data: Set[Tuple[str, str]]) -> Tuple[Set, Set, Dict]:
        role_set, filler_set, binding = set(), set(), defaultdict(list)
        for role, filler in frame_data:
            role_set.add(role)
            filler_set.add(filler)
            binding[role].append(filler)
        return role_set, filler_set, binding

    def _fuzzy_match(self, gold_list: StringList, pred_list: StringList) -> List[int]:
        match_matrix = gold_list @ pred_list
        max_indices = np.argmax(match_matrix, axis=0)
        valid_indices = [row
                         for col, row in enumerate(max_indices)
                         if match_matrix[row, col] == 100
                         ]
        return valid_indices

    def _intersect_between_binding(self, gold: Dict, pred: Dict) -> int:
        # Index the roles in gold first
        gold_roles = {idx: role for idx, role in enumerate(gold.keys())}

        # Fuzzy match the roles in pred with the roles in gold
        gold_array = StringList(list(gold_roles.values()))
        pred_array = StringList(list(pred.keys()))
        matched_roles = [gold_roles[idx] for idx in self._fuzzy_match(gold_array, pred_array)]

        # Fuzzy match the fillers between gold and pred sharing the same role
        num_intersect = 0
        for role in matched_roles:
            gold_fillers, pred_fillers = StringList(gold[role]), StringList(pred[role])
            num_intersect += len(self._fuzzy_match(gold_fillers, pred_fillers))

        return num_intersect

    def _union_between_binding(self, gold: Dict, pred: Dict) -> int:
        gold_bindings = [(role, f) for role, fillers in gold.items() for f in fillers]
        pred_bindings = [(role, f) for role, fillers in pred.items() for f in fillers]
        return len(set(gold_bindings).union(set(pred_bindings)))

    def calculate(self) -> Dict[int, Tuple]:
        if self.frames:
            for frame, span in self.frames.items():
                gold_roles, gold_fillers, gold_binding = self._organize(self.gold[self.guid][span])
                pred_roles, pred_fillers, pred_binding = self._organize(self.pred[self.guid][frame])
                role_iou = self._iou(pred_roles, gold_roles)
                filler_iou = self._iou(pred_fillers, gold_fillers)
                binding_iou = self._intersect_between_binding(gold_binding, pred_binding) / self._union_between_binding(gold_binding, pred_binding)
                self.frame_score[frame] = (role_iou, filler_iou, binding_iou)
        else:
            logging.warning("No overlap frames are found between gold and prediction data")
            self.frame_score = {-1: (-1, -1, -1)}
        return self.frame_score

#--------------------------------------------------------------------
# Run evaluation in parallel
#--------------------------------------------------------------------
def _load_data_from_dir(directory: Union[str, os.PathLike], label: str) -> Dict[str, Dict]:
    """
    Load data from a directory

    :param: dir: the directory path of data
    :param: label: the label of either gold or prediction data
    :return: a dictionary whose key is a GUID and value is a dictionary of frame data
    """
    logging.debug("Start loading %s data:", label)
    out = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if label == 'gold':
                out.update(load_gold(os.path.join(root, file)))
            else:
                out.update(load_pred(os.path.join(root, file)))
    return out


def run_eval(gold_dir: Union[str, os.PathLike], pred_dir: Union[str, os.PathLike]) -> List[Dict[str, Dict[int, Tuple]]]:
    """
    Run evaluation in parallel

    :param: gold_dir: the directory path of gold standard data
    :param: pred_dir: the directory path of prediction data
    :return: a dictionary of evaluation results
    """
    def help_run_iou(args):
        g, p = args
        iou = IOU(g, p)
        return iou.calculate()

    golds, preds = _load_data_from_dir(gold_dir, 'gold'), _load_data_from_dir(pred_dir, 'pred')
    overlap_videos = list(golds.keys() & preds.keys())
    logging.debug("\nOverlap videos %s found", len(overlap_videos))

    results = []
    if overlap_videos:
        num_cores = mp.cpu_count()
        num_processes = max(1, num_cores // 2)  # Use half of the available cores at maximum
        logging.debug("Number of processes: %s deployed", num_processes)

        with mp.Pool(num_processes) as pool:
            chunk_size = len(overlap_videos) // num_processes
            for i in range(0, len(overlap_videos), chunk_size):
                chunk = [({guid: golds[guid]}, {guid: preds[guid]})
                         for guid in overlap_videos[i:i+chunk_size]
                         ]
                results.extend(pool.map(help_run_iou, chunk))
    logging.warning("No overlap videos found")
    return results

#--------------------------------------------------------------------
# Write out evaluation results
#--------------------------------------------------------------------
def write_out(results: List[Dict[str, Dict[int, Tuple]]]) -> None:
    """Write out a list of formatted evaluation results into a .txt

    :param: results: the list of formatted evaluation results
    :return: None
    """
    with open('results.txt', 'w', encoding='utf-8') as f:
        for result in results:
            guid, frame_scores = result
            f.write(f"{guid}:\n")
            for frame, scores in frame_scores.items():
                f.write(f"\t\t{frame}: Role={scores[0]}\tFiller={scores[1]}\tBinding={scores[2]}\n")
    f.close()


#--------------------------------------------------------------------
# Arguments setting
#--------------------------------------------------------------------
def parse_args() -> Namespace:
    """Provide arguments of the script
    """
    parser = ArgumentParser(
        description='Evaluation script for RFB (Role-Filler-Binding)'
    )

    parser.add_argument(
        '-p',
        '--preds',
        help='The directory path of RFB predictions',
        required=True
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Set the debug mode'
    )

    return parser.parse_args()


#--------------------------------------------------------------------
# Main
#--------------------------------------------------------------------
def main():
    """Main function for running the evaluation task for the RFB app
    """
    args = parse_args()
    preds_dir, debug = args.preds, args.debug

    if debug:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    try:
        golds_dir = goldretriever.download_golds(GOLD_URL)
    except FileNotFoundError:
        logging.error("The gold standard data is not found")
        return

    iou_results = run_eval(golds_dir, preds_dir)
    write_out(iou_results)

if __name__ == "__main__":
    main()
