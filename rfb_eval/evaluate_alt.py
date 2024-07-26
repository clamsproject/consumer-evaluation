"""Description
This script is to complete evaluation process for the app RFB(role-filler-binding).
Used evaluation metrics include:
    + Intersection Over Union (IOU)
    + Dice Sørensen Coefficient (DSC)
"""

import json
import os
from typing import Dict, Set, Optional, Tuple, Union, List
from collections import defaultdict
from io import StringIO
from argparse import ArgumentParser, Namespace

from mmif import Mmif, AnnotationTypes, View, Annotation
from mmif.utils import video_document_helper as vdh
from clams_utils.aapb import guidhandler

import pandas as pd

SWT_APP = 'http://apps.clams.ai/swt-detection/v5.0'
RFB_APP = 'http://apps.clams.ai/role-filler-binder/41cb5b8'

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

def get_aligned_ann_of(mmif: Mmif, source: Annotation, source_app: str, target_app: str) -> Optional[Annotation]:
    """
    Get the aligned annotation of the input annotation cross views in MMIF
    
    :param: mmif: the mmif file
    :param: source: the source annotation that looks for its aligned annotation in target view
    :param: source_app: the app that contains source annotation
    :param: target_app: the app that might contain target aligned annotation 
    
    :return: either None type or an annotation
    """
    valid_views = {view.metadata.app: view for view in mmif.views if not (view.has_error() or view.has_warnings())}

    # Validate if two apps are in mmif
    if not (source_app and target_app) in valid_views:
        raise ValueError(f"Either {source_app} or {target_app} is not in mmif")
    
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
    
    rfb_mmif = Mmif(json.load(open(file, encoding='utf-8')))
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
    GUID | FRAME | SKIPPED | ANNOTATIONS
    ------------------------------------
    str  | int   |   T/F   | csv-string
    """
    guid = guidhandler.get_aapb_guid_from(gold_csv)
    frames_dict = defaultdict(set)
    df = pd.read_csv(gold_csv).dropna(subset=['ANNOTATIONS'])  #FIXME: Empty annotations are dropped from this process
    for _, frame in df.iterrows():
        if not frame['SKIPPED']:
            frames_dict[frame['FRAME']] = csv_string_to_set(frame['ANNOTATIONS'])
    return {guid: frames_dict}

#--------------------------------------------------------------------
# Class of evaluation metrics
#--------------------------------------------------------------------
class RFBMetrics:
    """
    The class of evaluation metrics for evaluating RFB 
    """
    @staticmethod
    def iou(
        pred: Dict[str, Dict],
        gold: Dict[str, Dict],
        key: str
    ) -> Tuple[str, float, Dict[int, float]]:
        """
        Calculate the IOU value for a single frame on either role or filler 
        as well as macro-avg of each frame's IOU value
        
        :param: pred: the processed prediction data
        :param: gold: the processed gold data
        :param: key: the object to be calculated. here is either {role, filler}
        :return: a tuple formatted as (macro-avg, {frame_num: iou}) 
        """
        def _iou(p, g):
            try:
                val = len(set.intersection(p, g)) / len(set.union(p, g))
            except ZeroDivisionError:
                val = 0
            return val

        # Check if pred and gold data refer to the same video
        if list(pred.keys())[0] != list(gold.keys())[0]:
            raise ValueError('The prediction file and the gold file refer to different videos.')
        
        # Find intersected frames between pred and gold
        guid = next(iter(pred.keys()))  
        intersect_frames = set(pred[guid].keys(), gold[guid].keys())
        
        # Calculate the iou
        obj_map = {'role':0, 'filler': 1}
        frame_score = {}
        sum_of_iou = 0
        for frame in intersect_frames:
            set_in_pred = {pair[obj_map[key]] for pair in pred[guid][frame]}
            set_in_gold = {pair[obj_map[key]] for pair in gold[guid][frame]}
            iou_val = _iou(set_in_pred, set_in_gold)
            frame_score[frame] = iou_val
            sum_of_iou += iou_val

        return guid, sum_of_iou / len(intersect_frames), frame_score   
                
#--------------------------------------------------------------------
# Write out evaluation results
#--------------------------------------------------------------------
def write_out(results: List[Tuple[str, float, Dict[int, float]]]) -> None:
    with open('results.txt', 'w', encoding='utf-8') as f:
        for result in results:
            guid, macro_avg, frame_scores = result
            f.write(f"{guid}: {macro_avg}\n")
            for frame, score in frame_scores.items():
                f.write(f"\t\t{frame}: {score}")
    f.close()


#--------------------------------------------------------------------
# Arguments setting
#--------------------------------------------------------------------
def parse_args() -> Namespace:
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
        '-g',
        '--golds',
        help="The directory path of gold standard data",
        required=True
    )
    
    return parser.parse_args()


#--------------------------------------------------------------------
# Main 
#--------------------------------------------------------------------
def main():
    args = parse_args()
    preds_dir, golds_dir = args.preds, args.golds
    
    golds = {}
    for root, _, gold_files in os.walk(golds_dir):
        for gold_file in gold_files:
            golds.update(load_gold(os.path.join(root, gold_file)))
            
    preds = {}
    for root, _, pred_files in os.walk(preds_dir):
        for pred_file in pred_files:
            preds.update(load_pred(os.path.join(root, pred_file)))
    
    overlap_videos = set(golds.keys() & preds.keys())
    iou_results = [RFBMetrics.iou(preds[guid], golds[guid], 'role') for guid in overlap_videos]
    write_out(iou_results)

if __name__ == "__main__":
    main()
        
    
     
    