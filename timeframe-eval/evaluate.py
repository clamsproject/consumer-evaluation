"""

Example uses:

python evaluate.py \
    --mmif-dir preds@slatedetection@aapb-collaboration-7/ \
    --gold-dir downloaded-golds/slate \
    --out-dir results/slate-apb-collaboration-7 \
    --slate

python evaluate.py \
    --mmif-dir preds@chyron-detection@batch2/ \
    --out-dir results/chyron-batch2 \
    --chyron

python evaluate.py \
    --mmif-dir preds@chyron-detection@batch2/ \
    --gold-dir downloaded-golds/chyron \
    --out-dir results/chyron-batch2 \
    --chyron

python evaluate.py \
    --mmif-dir preds@swt@3.1@aapb-collaboration-7/ \
    --gold-dir downloaded-golds/swt \
    --out-dir results/swt-aapb-collaboration-7 \
    --swt

python evaluate.py \
    --mmif-dir preds@swt@3.1@batch2/ \
    --gold-dir downloaded-golds/swt \
    --out-dir results/swt-batch2 \
    --swt


TODO:
- change interface to make dowloading golds and then using them later easier
    (now it involves tracking down a tmp dir in /var)
- make it work for all categories

"""


import argparse
from collections import defaultdict, namedtuple
import csv
import math
from pathlib import Path
import sys

from mmif import Mmif, DocumentTypes, AnnotationTypes
from mmif.serialize.annotation import Document
from mmif.utils import timeunit_helper as tuh
from mmif.utils import video_document_helper as vdh
from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.detection import DetectionErrorRate, DetectionPrecisionRecallFMeasure

from clams_utils.aapb import goldretriever


DEFAULT_FRAME_RATE = 29.97

# The three cases we have so far, all referring to annotation tasks
CHYRON = 'chyron'
SLATE = 'slate'
SWT = 'swt'

# Labels defined for SWT, assumes a particular kind of post-binning...
SWT_LABELS = ['bars', 'chyron', 'credits', 'slate']

# ... and this is that mapping
BINS = { 'B': 'bars', 'S': 'slate', 'I': 'chyron', 'Y': 'chyron', 'C': 'credits' }

# Gold standard locations
BASE_URL = 'https://github.com/clamsproject/aapb-annotations/tree'
GOLD_URLS = {
    CHYRON: f"{BASE_URL}/cc0d58e16a06a8f10de5fc0e5333081c107d5937/newshour-chyron/golds",
    SLATE: f"{BASE_URL}/b1d6476b6be6f9ffcb693872931d4d40e84449c8/january-slates/golds",
    SWT: f"{BASE_URL}/73f6d308/scene-recognition/golds/timeframes" }

# Maximum files to take, for debugging, may be removed any time
MAX_FILES = sys.maxsize


ErrorRate = namedtuple('ErrorRate', ['guid', 'miss', 'false', 'total', 'rate'])


def download_gold_standard(label: str):
    """Download gold standards for slates, chyrons or scenes-with-text. Uses the
    goldretriever utility, which downloads the files to a new temporary 0directory
    in /var/folders and returns that directory."""
    print('>>> downloading gold standard')
    download_dir = goldretriever.download_golds(GOLD_URLS[label])
    print(f'>>> gold standard files were downloaded to "{download_dir}"')
    return download_dir


class Timeframes:

    """Abstract class that defines how timeframes are stored, which can be either
    from the gold standard or system predictions. The core data structure is the
    dictionary in self.timeframes, which is structured as follows:

    {
        aapb_guid_1 -> { label_a -> Timeline, label_b -> Timeline }, ...}
        aapb_guid_2 -> { ... },
        ...
    }
    """

    def __init__(self, task: str, directory: str, regular_expression: str):
        self.task = task
        self.filenames = list(Path(directory).glob(regular_expression))
        self.fname2guid = { f: get_guid_from_filename(f) for f in self.filenames }
        self.guids = set(self.fname2guid.values())
        self.timeframes = {}
        self.marked = set()
        self.exclude = set()

    def __str__(self):
        return f'<{self.__class__.__name__} for task "{task}" with {len(self)} guids>'

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def get_label(row, task):
        """Return the label from the row, if there is none then return the task name
        as the label. For the SWT task we use label binning, in the latter case it is
        possible that the label returned is None."""
        gold_label = row.get('type label')
        gold_label = task if gold_label is None else gold_label
        if task == SWT:
            gold_label = BINS.get(gold_label)
        return gold_label

    def timeline(self, guid: str, label: str) -> Timeline:
        """Return the timeline for the guid and the label."""
        return self.timeframes[guid].get(label, Timeline())

    def add_segment(self, aapb_guid, gold_label, start, end):
        segment = Segment(start, end)
        self.timeframes[aapb_guid].setdefault(task, Timeline()).add(segment)

    def pp(self):
        for guid in sorted(self.timeframes):
            for label in sorted(self.timeframes[guid]):
                timeline = self.timeframes[guid][label]
                marked = '*' if guid in self.marked else ' '
                print(f'{guid}{marked}  {task:8s}  -->  {len(timeline):3d} timeframes')

    def pp_pair(self, other):
        for guid in sorted(self.timeframes):
            for label in sorted(self.timeframes[guid]):
                timeline = self.timeline(guid, label)
                timeline_other = other.timeline(guid, label)
                print(f'{guid}  {task:8s}  ', end='')
                print(f'-->  {len(timeline):3d} and {len(timeline_other):3d} timeframes')


class GoldData(Timeframes):

    def __init__(self, gold_directory: str, task: str):
        print(f'>>> loading gold standard files for task "{task}" in "{gold_dir}"')
        super().__init__(task, gold_directory, '*.csv')
        for n, gold_fname in enumerate(sorted(self.filenames)):
            if n >= MAX_FILES:
                break
            self._add_file_data(gold_fname, task)

    def _add_file_data(self, gold_fname: Path, task: str):
        aapb_guid = gold_fname.stem
        self.timeframes[aapb_guid] = {}
        with open(gold_fname, 'r') as gold_file:
            for i, row in enumerate(csv.DictReader(gold_file, delimiter=',')):
                try:
                    start = tuh.convert(row['start'], 'iso', 'sec', 0)
                    end = tuh.convert(row['end'], 'iso', 'sec', 0)
                    gold_label = self.get_label(row, task)
                    if gold_label is not None:
                        self.add_segment(aapb_guid, gold_label, start, end)
                except ValueError:
                    print(f"Invalid time format in {gold_fname}@{i}: {row}")


class Predictions(Timeframes):

    def __init__(self, mmif_dir, task: str, frame_types: list):
        print(f'>>> loading predictions for task "{task}" and frameTypes={frame_types}')
        print(f'    loading predictions from directory "{mmif_dir}"')
        super().__init__(task, mmif_dir, '*.mmif')
        for n, mmif_file in enumerate(sorted(self.filenames)):
            if n >= MAX_FILES:
                break
            self._add_file_data(mmif_file, task, frame_types)

    def _add_file_data(self, mmif_file: Path, task: str, frame_types: list):
        mmif = Mmif(open(mmif_file).read())
        vd = get_video(mmif_file, mmif)
        if vd is None:
            return
        fps = get_frame_rate(vd)
        aapb_guid = Path(vd.location).stem
        self.timeframes[aapb_guid] = {}
        view = mmif.get_view_contains(AnnotationTypes.TimeFrame)
        if view is None:
            print(f"Warning: no TimeFrame view found in {mmif_file}")
            self.exclude.add(aapb_guid)
            return
        for tf_ann in view.get_annotations(AnnotationTypes.TimeFrame):
            frame_type = tf_ann.get_property('frameType')
            if frame_type in frame_types:
                tu = tf_ann.get_property('timeUnit')
                s = tuh.convert(mmif.get_start(tf_ann), tu, 'sec', fps)
                e = tuh.convert(mmif.get_end(tf_ann), tu, 'sec', fps)
                self.add_segment(aapb_guid, frame_type, s, e)


def synchronize(golds: GoldData, preds: Predictions) -> set:
    """For both data sets, mark the guids that occur in both of them. Returns the
    set of guids that occur in both."""
    intersection = golds.guids.intersection(preds.guids)
    print(f'Overlapping guids: {len(intersection)}')
    golds.marked = intersection
    preds.marked = intersection
    return intersection


def prune(golds: GoldData, preds: Predictions):
    """Mark the guids that occur in both datasets and then for each only keep the data
    for the guids that occur in the intersection."""
    intersection = synchronize(golds, preds)
    for guid in golds.exclude:
        if guid in intersection:
            intersection.remove(guid)
    for guid in preds.exclude:
        if guid in intersection:
            intersection.remove(guid)
    golds.filenames = [fn for fn in golds.filenames if fn.stem in intersection]
    preds.filenames = [fn for fn in preds.filenames if fn.stem in intersection]
    golds.timeframes = {
        guid: val for guid, val in golds.timeframes.items() if guid in intersection }
    preds.timeframes = {
        guid: val for guid, val in preds.timeframes.items() if guid in intersection }
    

def calculate_detection_metrics(golds: GoldData, preds: Predictions, result_path: str, label: str):
    print(f'>>> calculating metrics for label={label}')
    detectER = DetectionErrorRate()
    detectPR = DetectionPrecisionRecallFMeasure()
    stats = Stats()
    for guid in preds.timeframes:
        reference = Annotation()
        for segment in golds.timeline(guid, label):
            reference[segment] = "aapb"
        hypothesis = Annotation()
        for segment in preds.timeline(guid, label):
            hypothesis[segment] = "aapb"
        error_rate = detectER(reference, hypothesis, collar=1.0, detailed=True)
        pr_re_fm = detectPR.compute_components(reference, hypothesis, collar=1.0, detailed=True)
        true_positive = pr_re_fm['relevant retrieved']
        false_negative = pr_re_fm['relevant'] - true_positive
        false_positive = pr_re_fm['retrieved'] - true_positive
        stats.increment_tp(true_positive)
        stats.increment_fp(false_positive)
        stats.increment_fn(false_negative)
        #print('==', error_rate)
        #print('==', pr_re_fm)
        #print_results(true_positive, false_positive, false_negative, stats)
        stats.add_error(guid, error_rate)
    print_metrics(stats, result_path)


def generate_side_by_side(golds: GoldData, preds: Predictions, outdir: Path, label: str):
    print(f'>>> generating side-by-side in {outdir}')
    for guid in golds.timeframes:
        no_detection = False
        path = outdir / f"{guid}.sbs.csv"
        gold_time_chunks = []
        test_time_chunks = []
        for segment in golds.timeline(guid, label):
            if segment.end == 0:
                continue
            gold_start = math.floor(segment.start)
            gold_end = math.floor(segment.end)
            gold_time_chunks.extend(range(gold_start, gold_end + 1))
        if guid in preds.timeframes:
            for segment in preds.timeline(guid, label):
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


class Stats:

    def __init__(self, tp=0, fp=0, fn=0):
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.data = []

    def increment_tp(self, count):
        self.tp += count

    def increment_fp(self, count):
        self.fp += count

    def increment_fn(self, count):
        self.fn += count

    def precision(self):
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def fscore(self):
        p = self.precision()
        r = self.recall()
        return 0.0 if p + r == 0 else (2 * p * r) / (p + r)

    def add_error(self, guid: str, error_rate: dict):
        self.data.append(
            ErrorRate(
                guid, error_rate['miss'], error_rate['false alarm'],
                error_rate['total'], error_rate['detection error rate']))


def print_metrics(stats: Stats, outfile: str):
    p = stats.precision()
    r = stats.recall()
    f = stats.fscore()
    with open(outfile, 'w') as fh:
        fh.write(f'Total Precision = {p:.2f}\n')
        fh.write(f'Total Recall    = {r:.2f}\n')
        fh.write(f'Total F1        = {f:.2f}\n')
        fh.write(f'\n\nIndividual file results:\n\n')
        fh.write(f'                                  miss   false   total   rate\n')
        for n, error_rate in enumerate(stats.data):
            fh.write(f'{n:3d}  {error_rate.guid:>25}  {error_rate.miss:6.2f}')
            fh.write(f'  {error_rate.false:6.2f}  {error_rate.total:6.2f}')
            fh.write(f'  {error_rate.rate:5.2f}\n')
    print(f'>>> metrics: p={p:.2f} r={r:.2f} f={f:.2f}')
    print(f'>>> wrote metrics to {outfile}')


def print_results(true_positive, false_positive, false_negative, stats):
    """Debugging method."""
    print(f"-- tpfpfn  tp={true_positive:.4f} fp={false_positive:.4f} fn={false_negative:.4f}")
    print(f"-- TDFPFN  TP={stats.tp:.4f} FP={stats.fp:.4f} FN={stats.fn:.4f}\n")


def get_task(args):
    """Returns a string representation of the task label, which indicates the kind of
    annotations we are evaluating. This replaces the various booleans that determine
    the label type with a single string, and simplifies things downstream."""
    if args.chyron:
        return CHYRON
    elif args.slate:
        return SLATE
    elif args.swt:
        return SWT
    else:
        return None


def get_video(mmif_file: Path, mmif: Mmif) -> Document:
    """Return a Document or """
    vds = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
    if vds:
        return vds[0]
    else:
        sys.stderr.write(f"No video document found in {mmif_file}\n")
        return None


def get_frame_rate(vd):
    """Return the video's framerate."""
    # TODO: this used to be done by using the helper, but that was replaced by a
    # hard-coded value, not sure why that was done (MV)
    # fps = vdh.get_framerate(vd)
    return DEFAULT_FRAME_RATE


def get_guid_from_filename(fpath: Path) -> str:
    guid = fpath.stem
    if guid.endswith('.slatedetection'):
        guid = guid[:-15]
    return guid


def parse_arguments():
    parser = argparse.ArgumentParser(description='Timeframe evaluation')
    parser.add_argument('--mmif-dir', help='directory containing machine annotated files (MMIF)', required=True)
    parser.add_argument('--out-dir', help='directory to publish metrics and side-by-side results', required=True)
    parser.add_argument('--gold-dir', help='directory with gold standard files')
    gold_group = parser.add_mutually_exclusive_group(required=True)
    gold_group.add_argument('--slate', action='store_true', help='slate annotations')
    gold_group.add_argument('--chyron', action='store_true', help='chyron annotations')
    gold_group.add_argument('--swt', action='store_true', help='SWT annotations')
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_arguments()
    task = get_task(args)
    frame_types = SWT_LABELS if task == SWT else [task]

    gold_dir = args.gold_dir if args.gold_dir else download_gold_standard(task)
    if gold_dir is None:
        raise ValueError("No gold standard provided")
    gold_dir = Path(gold_dir)
    out_dir = Path(args.out_dir)
    sbs_dir = out_dir / 'sbs'
    sbs_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f'metrics.txt'

    gold_timeframes = GoldData(gold_dir, task)
    pred_timeframes = Predictions(args.mmif_dir, task, frame_types)
    print(gold_timeframes)
    print(pred_timeframes)
    prune(gold_timeframes, pred_timeframes)
    #gold_timeframes.pp_pair(pred_timeframes)

    calculate_detection_metrics(gold_timeframes, pred_timeframes, metrics_file, task)
    generate_side_by_side(gold_timeframes, pred_timeframes, sbs_dir, task)
