"""

Example uses:

python evaluate.py \
    --mmif-dir preds@slatedetection@aapb-collaboration-7/ \
    --side-by-side results/sbs-slate \
    --result-file results/results-slate.txt \
    --gold-dir downloaded-golds/slate \
    --slate

python evaluate.py \
    --mmif-dir preds@chyron-detection@batch2/ \
    --side-by-side results/sbs-chyron \
    --result-file results/results-chyron.txt \
    --chyron

python evaluate.py \
    --mmif-dir preds@chyron-detection@batch2/ \
    --side-by-side results/sbs-chyron \
    --result-file results/results-chyron.txt \
    --gold-dir downloaded-golds/chyron \
    --chyron

python evaluate.py \
    --mmif-dir  preds@swt@3.1@aapb-collaboration-7/ \
    --side-by-side results/sbs-swt \
    --result-file results/results-swt.txt \
    --gold-dir downloaded-golds/swt \
    --swt


TODO:
- change interface to make dowloading golds and then using them later easier
    (now it involves tracking down a tmp dir in /var)
- make it work for all categories

"""


import argparse
import collections
import csv
import math
import pathlib
import sys

import pandas as pd
from mmif import Mmif, DocumentTypes, AnnotationTypes
from mmif.serialize.annotation import Document
from mmif.utils import timeunit_helper as tuh
from mmif.utils import video_document_helper as vdh
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecisionRecallFMeasure

from clams_utils.aapb import goldretriever


# The three cases we have so far
CHYRON = 'chyron'
SLATE = 'slate'
SWT = 'swt'

# Labels defined for SWT
SWT_LABELS = ['bars', 'chyron', 'credits', 'slate']

# Gold standard locations
BASE_URL = 'https://github.com/clamsproject/aapb-annotations/tree'
GOLD_URLS = {
    CHYRON: f"{BASE_URL}/cc0d58e16a06a8f10de5fc0e5333081c107d5937/newshour-chyron/golds",
    SLATE: f"{BASE_URL}/b1d6476b6be6f9ffcb693872931d4d40e84449c8/january-slates/golds",
    SWT: f"{BASE_URL}/73f6d308/scene-recognition/golds/timeframes" }


def download_gold_standard(label: str):
    """Download gold standards for slates, chyrons or scenes-with-text. Uses the
    goldretriever utility, which downloads the files to a new temporary 0directory
    in /var/folders and returns that directory."""
    print('>>> downloading gold standard')
    ref_dir = goldretriever.download_golds(GOLD_URLS[label])
    print(f'>>> gold standard files were downloaded to "{ref_dir}"')
    return ref_dir


def load_gold_standard(gold_dir: str, label: str) -> dict:
    print(f'>>> loading gold standard files in "{gold_dir}"')
    gold_timeframes = {}
    for gold_fname in sorted(pathlib.Path(gold_dir).glob("*.csv")):
        # TODO: for now only taking the first five files
        # TODO: to be deleted
        if (gold_fname.name == 'cpb-aacip-507-028pc2tp2z.csv'
            or gold_fname.name == 'cpb-aacip-507-9882j68s35.csv'
            or gold_fname.name == 'cpb-aacip-129-00ns1w7z.csv'):
            break
        with open(gold_fname, 'r') as gold_file:
            aapb_guid = gold_fname.stem
            r = csv.DictReader(gold_file, delimiter=',')
            for i, row in enumerate(r):
                #print(i, row, row.get('type label'))
                try:
                    start = tuh.convert(row['start'], 'iso', 'sec', 0)
                    end = tuh.convert(row['end'], 'iso', 'sec', 0)
                    type_label = row.get('type label')
                    label = type_label if type_label else label
                    add_segment(gold_timeframes, aapb_guid, label, start, end)
                except ValueError:
                    sys.stderr.write(f"Invalid time format in {gold_fname}: {row} @ {i}\n")
    #print_timeframes(gold_timeframes)
    return gold_timeframes


def add_segment(gold_timeframes: dict, aapb_guid: str, label, start: float, end: float):
    segment = Segment(start, end)
    default_dict = collections.defaultdict(Timeline)
    gold_timeframes.setdefault(label, default_dict)[aapb_guid].add(segment)


def process_mmif_files(mmif_dir, gold_timeframes: dict, label, frame_types):
    print(f'>>> loading predictions for label "{label}"')
    mmif_files = pathlib.Path(mmif_dir).glob("*.mmif")
    pred_timeframes_new = {}
    pred_timeframes = collections.defaultdict(Timeline)

    #gold_timeframes =  gold_timeframes[label]
    for mmif_file in sorted(mmif_files):
        mmif = Mmif(open(mmif_file).read())
        vd = get_video(mmif_file, mmif)
        if vd is None:
            continue
        aapb_guid = pathlib.Path(vd.location).stem
        if aapb_guid in gold_timeframes:
            print(f'{mmif_file} - processing {label}')
            v = mmif.get_view_contains(AnnotationTypes.TimeFrame)
            if v is None:
                sys.stderr.write(f"No TimeFrame view found in {mmif_file}\n")
                continue
            for tf_ann in v.get_annotations(AnnotationTypes.TimeFrame):
                if not tf_ann.get_property('frameType') in frame_types:
                    continue
                # fps = vdh.get_framerate(vd)
                # TODO: this should not be hardcoded like this
                fps = 29.97
                tu = tf_ann.get_property('timeUnit')
                s = mmif.get_start(tf_ann)
                e = mmif.get_end(tf_ann)
                segment = Segment(*(tuh.convert(t, tu, 'sec', fps) for t in (s, e)))
                pred_timeframes[aapb_guid].add(segment)
    print('>>> printing predictions')
    print_timeframes_for_label(pred_timeframes, label=label)
    return pred_timeframes


# adapt the code from Kelley Lynch - 'evaluate_chyrons.py'
# add the situation which the mmif file is not in the gold timeframes
def calculate_detection_metrics(gold_timeframes_dict, pred_timeframes, result_path, label):
    print('>>> calculating metrics')
    gold_timeframes_dict =  gold_timeframes_dict[label]
    detectER = DetectionErrorRate()
    detectPR = DetectionPrecisionRecallFMeasure()
    TP = 0
    FP = 0
    FN = 0
    data = pd.DataFrame(columns=['GUID', 'FN seconds', 'FP seconds', 'Total true seconds'])
    for file_ID in pred_timeframes:
        reference = Annotation()
        for segment in gold_timeframes_dict[file_ID]:
            reference[segment] = "aapb"
        hypothesis = Annotation()
        for segment in pred_timeframes[file_ID]:
            hypothesis[segment] = "aapb"
        results_dict = detectER.compute_components(reference, hypothesis, collar=1.0, detailed=True)
        average = detectPR.compute_components(reference, hypothesis, collar=1.0, detailed=True)
        true_positive = average['relevant retrieved']
        false_negative = average['relevant'] - true_positive
        false_positive = average['retrieved'] - true_positive
        TP += true_positive
        FP += false_positive
        FN += false_negative
        #print_results(results_dict, average,
        #              true_positive, false_positive, false_negative, TP, FP, FN)
        data = pd.concat(
            [
                data,
                pd.DataFrame({'GUID': file_ID,
                              'FN seconds': results_dict['miss'],
                              'FP seconds': results_dict['false alarm'],
                              'Total true seconds': results_dict['total']},
                             index=[0])
            ], ignore_index=True)
    p = precision(TP, FP)
    r = recall(TP, FN)
    f = fscore(p, r)
    with open(result_path, 'w') as out_f:
        out_f.write(f'Total Precision = {p:.2f}\n')
        out_f.write(f'Total Recall    = {r:.2f}\n')
        out_f.write(f'Total F1        = {f:.2f}\n')
        out_f.write(f'\n\nIndividual file results: \n')
        out_f.write(data.to_string(index=True))


def generate_side_by_side(
        gold_timeframes: dict,
        pred_timeframes: collections.defaultdict,
        outdir: pathlib.Path,
        label: str):
    print('>>> generating side-by-side')
    #for par in gold_timeframes, pred_timeframes, outdir:
    #    print(f'--- PAR: {type(par)} {str(par)[:100]}')
    gold_timeframes =  gold_timeframes[label]
    for guid in gold_timeframes:
        no_detection = False
        path = outdir / f"{guid}.sbs.csv"
        gold_time_chunks = []
        test_time_chunks = []
        for segment in gold_timeframes[guid]:
            if segment.end == 0:
                continue
            gold_start = math.floor(segment.start)
            gold_end = math.floor(segment.end)
            gold_time_chunks.extend(range(gold_start, gold_end + 1))
        if guid in pred_timeframes:
            for segment in pred_timeframes[guid]:
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
                    gold = 1 if i in gold_time_chunks else 0
                    test = 1 if i in test_time_chunks else 0
                    out_f.write(",".join([interval, str(gold), str(test)]))
                    out_f.write("\n")
                    i += 1
                #for i in range(maximum)
                #print(range(maximum))


def precision(true_positives: int, false_positives: int):
    try:
        return true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        return 0.0

def recall(true_positives: int, false_negatives: int):
    try:
        return true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        return 0.0

def fscore(precision, recall):
    if precision + recall == 0:
        return 0.0
    else:
        return (2 * precision * recall) / (precision + recall)


def print_timeframes(gold_timeframes: dict, verbose=False):
    """The input dictionary has labels as keys, the valuas are dictionaries with
    gids as keys and lists of TimeFrames as values."""
    for label in gold_timeframes:
        print_timeframes_for_label(gold_timeframes[label], label=label)

def print_timeframes_for_label(gold_timeframes: dict, verbose=False, label=None):
    """The input dictionary has guids as the keys and lists of TimeFramesas the
    values."""
    print(f'LABEL: {label}')
    for guid in sorted(gold_timeframes):
        segments = gold_timeframes[guid]
        print(f'    {guid}  -->  {len(segments):>3d} timeframes')
        if verbose:
            for n, segment in enumerate(segments):
                    print(f'        {n+1:2}  {segment}')

def print_results(results_dict, average_dict,
                  true_positive, false_positive, false_negative, TP, FP, FN):
    """Debugging method."""
    print('-- results', ' '.join(f'{k}={v:.4f}' for k, v in results_dict.items()))
    print('-- average', ' '.join(f'{k}={v:.4f}' for k, v in average_dict.items()))
    print(f"-- tpfpfn  tp={true_positive:.4f} fp={false_positive:.4f} fn={false_negative:.4f}}}")
    print(f"-- TDFPFN  TP={TP:.4f} FP={FP:.4f} FN={FN:.4f}}}\n")


def get_label(args):
    """Just replacing the various booleans that determine the label type with a
    single string, for ease of use downstream."""
    if args.chyron:
        return CHYRON
    elif args.slate:
        return SLATE
    elif args.swt:
        return SWT
    else:
        return None


def get_video(mmif_file: pathlib.Path, mmif: Mmif) -> Document:
    """Return a Document or """
    vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
    if vds:
        #print(type(vds[0]))
        return vds[0]
    else:
        sys.stderr.write(f"No video document found in {mmif_file}\n")
        return None


if __name__ == "__main__":

    # get the absolute path of video-file-dir and hypothesis-file-dir
    parser = argparse.ArgumentParser(description='Process some directories.')
    parser.add_argument('-m', '--mmif-dir', type=str, required=True,
                        help='directory containing machine annotated files (MMIF)')
    parser.add_argument('-s', '--side-by-side', help='directory to publish side-by-side results', default=None)
    parser.add_argument('-r', '--result-file', help='file to store evaluation results', default='results.txt')
    parser.add_argument('-g', '--gold-dir', help='file to store gold standard', default=None)
    gold_group = parser.add_mutually_exclusive_group(required=True)
    gold_group.add_argument('--slate', action='store_true', help='slate annotations')
    gold_group.add_argument('--chyron', action='store_true', help='chyron annotations')
    gold_group.add_argument('--swt', action='store_true', help='SWT annotations')
    args = parser.parse_args()

    # this label indicates what kind of annotations we are evaluating
    label = get_label(args)

    if args.side_by_side:
        outdir = pathlib.Path(args.side_by_side)
    else:
        outdir = pathlib.Path(__file__).parent

    ref_dir = args.gold_dir if args.gold_dir else download_gold_standard(label)
    if ref_dir is None:
        raise ValueError("No gold standard provided")
    ref_dir = pathlib.Path(ref_dir)

    gold_timeframes = load_gold_standard(ref_dir, label)
    
    # get the timeframes from the system predictions
    frame_types = SWT_LABELS if label == SWT else [label]
    pred_timeframes = process_mmif_files(
        args.mmif_dir, gold_timeframes, label, frame_types=frame_types)

    # make sure output directories exist
    outdir.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)

    # final calculation
    calculate_detection_metrics(gold_timeframes, pred_timeframes, args.result_file, label)
    generate_side_by_side(gold_timeframes, pred_timeframes, outdir, label)

    print("Done!")
