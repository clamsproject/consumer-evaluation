import argparse
import collections
import csv
import math
import pathlib
import sys
import re

import pandas as pd
from mmif import Mmif, DocumentTypes, AnnotationTypes
from mmif.utils import timeunit_helper as tuh
from mmif.utils import video_document_helper as vdh
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecisionRecallFMeasure

import goldretriever

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils import standardized_parser

# Constants
GOLD_CHYRON_URL = "https://github.com/clamsproject/aapb-annotations/tree/cc0d58e16a06a8f10de5fc0e5333081c107d5937/newshour-chyron/golds"
GOLD_SLATES_URL = "https://github.com/clamsproject/aapb-annotations/tree/b1d6476b6be6f9ffcb693872931d4d40e84449c8/january-slates/golds"

def load_gold_standard(gold_dir):
    gold_timeframes = collections.defaultdict(Timeline)
    for gold_fname in pathlib.Path(gold_dir).glob("*.csv"):
        with open(gold_fname, 'r') as gold_file:
            # aapb_guid = gold_fname.stem
            r = csv.DictReader(gold_file, delimiter=',')
            for i, row in enumerate(r):
                try:
                    start, end = (tuh.convert(re.sub(r'[^\d:.]', '', row[time_key].replace(";", ".")), 'iso', 'ms', 0)/1000 for time_key in ["Slate Start ,", "Slate End   ,"])
                    aapb_guid = row['GUID']
                    gold_timeframes[aapb_guid].add(Segment(start, end))
                except ValueError:
                    sys.stderr.write(f"Invalid time format in {gold_fname}: {row} @ {i}\n")

    return gold_timeframes


def process_mmif_file(mmif_dir, gold_timeframe_dict, frame_types):
    mmif_files = pathlib.Path(mmif_dir).glob("*.mmif")
    pred_timeframes = collections.defaultdict(Timeline)
    for mmif_file in mmif_files:
        mmif = Mmif(open(mmif_file).read())
        vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if vds:
            vd = vds[0]
        else:
            sys.stderr.write(f"No video document found in {mmif_file}\n")
            continue
        aapb_guid = pathlib.Path(vd.location).stem
        if aapb_guid in gold_timeframe_dict:
            print(f"evaluating {aapb_guid}...")
            v = mmif.get_view_contains(AnnotationTypes.TimeFrame)
            if v is None:
                sys.stderr.write(f"No TimeFrame found in {mmif_file}\n")
                continue
            for tf_ann in v.get_annotations(AnnotationTypes.TimeFrame):
                if not tf_ann.get_property('frameType') in frame_types:
                    continue
                # fps = vdh.get_framerate(vd)
                fps = 29.97
                tu = tf_ann.get_property('timeUnit')
                s = mmif.get_start(tf_ann)
                e = mmif.get_end(tf_ann)
                pred_timeframes[aapb_guid].add(Segment(*(tuh.convert(t, tu, 'sec', fps) for t in (s, e))))
    return pred_timeframes


# adapt the code from Kelley Lynch - 'evaluate_chyrons.py'
# add the situation which the mmif file is not in the gold timeframes
def calculate_detection_metrics(gold_timeframes_dict, test_timeframes, result_path):
    metric = DetectionErrorRate()
    final = DetectionPrecisionRecallFMeasure()
    TP = 0
    FP = 0
    FN = 0
    data = pd.DataFrame(columns=['GUID', 'FN seconds', 'FP seconds', 'Total true seconds'])
    for file_ID in test_timeframes:
        reference = Annotation()
        for segment in gold_timeframes_dict[file_ID]:
            reference[segment] = "aapb"
        hypothesis = Annotation()
        for segment in test_timeframes[file_ID]:
            hypothesis[segment] = "aapb"
        results_dict = metric.compute_components(reference, hypothesis, collar=1.0, detailed=True)
        average = final.compute_components(reference, hypothesis, collar=1.0, detailed=True)
        true_positive = average['relevant retrieved']
        false_negative = average['relevant'] - true_positive
        false_positive = average['retrieved'] - true_positive
        TP += true_positive
        FP += false_positive
        FN += false_negative
        data = pd.concat(
            [
                data,
                pd.DataFrame({'GUID': file_ID, 
                              'FN seconds': results_dict['miss'], 
                              'FP seconds': results_dict['false alarm'], 
                              'Total true seconds': results_dict['total']}, 
                             index=[0])
            ], ignore_index=True)
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    with open(result_path, 'w') as out_f:
        out_f.write(f'Total Precision = {str(precision)}\t Total Recall = {str(recall)}\t Total F1 = {str(f1)}\n\n\nIndividual file results: \n')
        out_f.write(data.to_string(index=True))


def generate_side_by_side(golddir, testdir, outdir):
    for guid in golddir:
        no_detection = False
        path = outdir / f"{guid}.sbs.csv"
        gold_time_chunks = []
        test_time_chunks = []
        for segment in golddir[guid]:
            if segment.end == 0:
                continue
            gold_start = math.floor(segment.start)
            gold_end = math.floor(segment.end)
            gold_time_chunks.extend(range(gold_start, gold_end + 1))
        if guid in testdir:
            for segment in testdir[guid]:
                test_start = math.floor(segment.start)
                test_end = math.ceil(segment.end)
                test_time_chunks.extend(range(test_start, test_end))
        if len(gold_time_chunks) > 0 and len(test_time_chunks) > 0:
            maximum = max(max(gold_time_chunks), max(test_time_chunks))
        elif len(gold_time_chunks) > 0:
            maximum = max(gold_time_chunks)
        elif len(test_time_chunks) > 0:
            maximum = max(test_time_chunks)
        else:
            no_detection = True
        with open(path, "w") as out_f:
            if no_detection:
                out_f.write("no timeframes annotated in gold or predicted by app")
            else:
                i = 0
                while i < maximum + 1:
                    interval = "(" + str(i) + " - " + str(i + 1) + ")"
                    if i in gold_time_chunks:
                        gold = 1
                    else:
                        gold = 0
                    if i in test_time_chunks:
                        test = 1
                    else:
                        test = 0
                    out_f.write(",".join([interval, str(gold), str(test)]))
                    out_f.write("\n")
                    i += 1


if __name__ == "__main__":
    # get the absolute path of video-file-dir and hypothesis-file-dir
    args = standardized_parser.parse_args()

    if args.side_by_side:
        outdir = pathlib.Path(args.side_by_side)
        if not outdir.exists():
            outdir.mkdir()
    else:
        outdir = pathlib.Path(__file__).parent

    ref_dir = None
    if args.gold_file:
        ref_dir = args.gold_file
    else:
        if args.slate:
            ref_dir = goldretriever.download_golds(GOLD_SLATES_URL)
        elif args.chyron:
            ref_dir = goldretriever.download_golds(GOLD_CHYRON_URL)
    if ref_dir is None:
        raise ValueError("No gold standard provided")
    ref_dir = pathlib.Path(ref_dir)

    gold_timeframes_dict = load_gold_standard(ref_dir)

    # create the 'test_timeframes'
    test_timeframes = process_mmif_file(args.pred_file, gold_timeframes_dict, frame_types=['slate' if args.slate else 'chyron' if args.chyron else ''])

    # final calculation
    calculate_detection_metrics(gold_timeframes_dict, test_timeframes, args.result_file)
    generate_side_by_side(gold_timeframes_dict, test_timeframes, outdir)

    print("Done!")

