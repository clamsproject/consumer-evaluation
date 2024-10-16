import argparse
import collections
import logging
import re
from pathlib import Path as P

import pandas as pd
import pyannote.metrics.base
from clams_utils.aapb import goldretriever, guidhandler
from lapps.discriminators import Uri
from mmif.serialize import Mmif
from mmif.vocabulary import AnnotationTypes
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationCoverage, DiarizationPurity
from pyannote.metrics.segmentation import SegmentationCoverage, SegmentationRecall, SegmentationPrecision, \
    SegmentationPurity

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils import standardized_parser, basic_eval

DEFAULT_GOLD_URL = 'https://github.com/clamsproject/aapb-annotations/tree/f884e10d0b9d4b1d68e294d83c6e838528d2c249/newshour-transcript-sync/golds/aapb-collaboration-21'

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)-8s %(thread)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger('forced-aligner-evaluator')


def read_cadet_annotation_tsv(tsv_file_list):
    def cadettime_to_ms(time_str):
        try:
            h, m, s = time_str.split(':')
            s, ms = s.split('.')
            return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
        except ValueError:
            raise ValueError("Invalid time format. Expected format: hh:mm:ss.mmm")

    gold_timeframes = collections.defaultdict(Annotation)
    for tsv_file in tsv_file_list:
        guid = guidhandler.get_aapb_guid_from(tsv_file.stem)
        df = pd.read_csv(tsv_file, sep='\t')
        for index, row in df[['starts', 'ends', 'content']].iterrows():
            segment = Segment(cadettime_to_ms(row['starts']) / 1000, cadettime_to_ms(row['ends']) / 1000)
            gold_timeframes[guid][segment] = row['content']
    return gold_timeframes


def read_system_mmif(mmif_file_list, reference_timeframes):
    def tokenize_cadet_silver_text(text):
        # kaldi/gentle does not return the same text as the original and 
        # does its own normalization/tokenization. This function tries to
        # reverse-engineer the original text from the kaldi/gentle output
        logger.debug(f'original: {text}')
        from string import punctuation
        # text = text.replace("%", " percent")
        # text = text.replace("’", "'")
        text = text.replace("`", " ")  # that`s
        text = text.replace("&", " ")  # AT&T
        text = text.replace("--", " ")
        text = text.replace("..", " ")
        text = text.replace("...", " ")
        # text = text.replace("'re", " re")
        punc_tokens = []
        tokens = []
        for token in text.split():
            # just skip single letter punctuations
            if not token or token in punctuation:
                continue
            # o'clock
            if re.search(r'\d\:\d\d', token):
                punc_tokens.extend(token.split(':'))
            # fractions
            elif re.match(r'\d+\/\d+', token):
                punc_tokens.extend(token.split('/'))
            # decimal numbers
            elif re.search(r'\d\.\d', token):
                punc_tokens.extend(token.split('.'))
            # comma separated large numbers
            elif re.search(r'\d,\d', token):
                punc_tokens.extend(token.split(','))
            # abbreviations (U.S.)
            elif re.match(r'[A-Z]\.([A-Z]\.?)+', token):
                punc_tokens.extend(token.split('.'))
            else:
                # hyphenated words
                punc_tokens.extend(token.split('-'))
        for token in punc_tokens:
            if re.match(r'\'[0-9][0-9]s?', token):
                while token[-1] in punctuation:
                    token = token[:-1]
                tokens.append(token)
            else:
                while token and token[-1] in punctuation:
                    token = token[:-1]
                while token and token[0] in punctuation:
                    token = token[1:]
                if token:
                    tokens.append(token)
        logger.debug(f'normaliz: {tokens}')
        return tokens

    test_timeframes = collections.defaultdict(Annotation)
    for mmif_file in mmif_file_list:
        guid = guidhandler.get_aapb_guid_from(mmif_file.stem)
        reference = reference_timeframes[guid]
        with open(mmif_file, 'r') as file:
            mmif = Mmif(file.read())
            ref_segs = reference.itertracks(yield_label=True)
            ref_segment_text = tokenize_cadet_silver_text(next(ref_segs)[2])
            in_segment = False
            view = mmif.get_view_contains(Uri.TOKEN)
            if view is None:
                continue
            timeunit = view.metadata.contains[AnnotationTypes.TimeFrame]['timeUnit']
            t2tf_alignments = {}
            for alignment in view.get_annotations(AnnotationTypes.Alignment):
                s = view[alignment.get_property('source')]
                t = view[alignment.get_property('target')]
                if s.at_type == Uri.TOKEN and t.at_type == AnnotationTypes.TimeFrame:
                    t2tf_alignments[s.id] = t
                elif t.at_type == Uri.TOKEN and s.at_type == AnnotationTypes.TimeFrame:
                    t2tf_alignments[t.id] = s
            s, e = 0, 0
            hyp_length = 0
            for i, token in enumerate(view.get_annotations(Uri.TOKEN)):
                # because gentle app returns disfluency tokens that are not in the reference without start/end
                if 'start' not in token.properties:
                    continue
                if i < 300000000:
                    first = ref_segment_text[0]
                    final = ref_segment_text[-1]
                    w = token.get_property('word')
                    logger.debug(' '.join(map(str, (token.id, in_segment, w,
                                                    first, w.lower() == first.lower(),
                                                    final, w.lower() == final.lower()))))
                hyp_length += 1
                if token.id in t2tf_alignments:
                    s = t2tf_alignments[token.id].get_property('start')
                    e = t2tf_alignments[token.id].get_property('end')
                # just in case gentle (kaldi) removed the first token during alignment 
                # (happens with some stopwords and symbols)
                if not in_segment and (token.get_property('word').lower() == ref_segment_text[0].lower()
                                       or token.get_property('word').lower() == ref_segment_text[1].lower()):
                    in_segment = True
                    start = s / 1000 if timeunit.startswith('mill') else s
                if in_segment and token.get_property('word').lower() == ref_segment_text[-1].lower():
                    end = e / 1000 if timeunit.startswith('mill') else e
                    test_timeframes[guid][Segment(start, end)] = ' '.join(ref_segment_text)
                    try:
                        ref_segment_text = tokenize_cadet_silver_text(next(ref_segs)[2])
                        hyp_length = 0
                    except StopIteration:
                        break
                    in_segment = False
            logger.debug(f'system token iteration is done for {guid}, last timeframe {start}, {e}')
            end = e / 1000 if timeunit.startswith('mill') else e
            test_timeframes[guid][Segment(start, end)] = ' '.join(ref_segment_text)
    return test_timeframes


def calculate_detection_metrics(gold_timeframes, test_timeframes, result_path, thresholds=[]):
    coverage = DiarizationCoverage()
    purity = DiarizationPurity()
    scoverage = SegmentationCoverage()
    spurity = SegmentationPurity()
    precision = SegmentationPrecision()
    recall = SegmentationRecall()

    # final = SegmentationCoverage()
    data = []
    for guid in gold_timeframes.keys():
        reference = gold_timeframes[guid]
        hypothesis = test_timeframes[guid]
        if len(hypothesis) == 0:
            logger.warning(f'{guid} :: no hypothesis found')
            continue
        if len(reference) != len(hypothesis):
            logger.warning(
                f'{guid} :: reference ({len(reference)}) and hypothesis ({len(hypothesis)}) have different number of segments')
        try:
            # coverage, purity, f1, precision, recall, f1
            cpfprf = [guid]
            for met in coverage, purity:
                comps = met.compute_components(reference, hypothesis)
                cpfprf.append(met.compute_metric(comps))
            cpfprf.append(pyannote.metrics.base.f_measure(cpfprf[-2], cpfprf[-1]))
            for met in scoverage, spurity:
                comps = met.compute_components(reference, hypothesis)
                cpfprf.append(met.compute_metric(comps))
            cpfprf.append(pyannote.metrics.base.f_measure(cpfprf[-2], cpfprf[-1]))
            for threshold in thresholds:
                for met in precision, recall:
                    comps = met.compute_components(reference, hypothesis, tolerance=threshold)
                    cpfprf.append(met.compute_metric(comps))
                cpfprf.append(pyannote.metrics.base.f_measure(cpfprf[-2], cpfprf[-1]))
            data.append(cpfprf)
        except KeyError:
            print(f"Error: Issue with keys in results for file {guid}")
    cols = ['GUID']
    cov_pur_colnames = 'DiaCov DiaPur D-C-P-F1 SegCov SegPur S-C-P-F1'.split()
    cols.extend(cov_pur_colnames)
    for threshold in thresholds:
        cols.extend(f'Precision@{threshold} Recall@{threshold} P-R-F1@{threshold}-sec-tolerance'.split())
    data = pd.DataFrame(data, columns=cols)
    res_str = (f'Individual file results:\n{data.to_string(index=False)}\n\n\n'
               f'Average results:\n{data.loc[:, data.columns != "GUID"].mean(axis=0)}')

    return res_str
    # print(res_str)
    # if result_path is not None:
    #     with open(result_path, 'w') as fh_out:
    #         fh_out.write(res_str)


if __name__ == "__main__":
    args = standardized_parser.parse_args()

    if args.gold_file is None:
        args.gold_file = goldretriever.download_golds(DEFAULT_GOLD_URL)
    gold_timeframes = read_cadet_annotation_tsv(P(args.gold_file).glob("*.tsv"))
    test_timeframes = read_system_mmif(P(args.pred_file).glob("*.mmif"), gold_timeframes)
    threshold = []
    for t in args.thresholds.split(','):
        if t:
            threshold.append(float(t))
    data = calculate_detection_metrics(gold_timeframes, test_timeframes, args.result_file, threshold)

    evaluator = basic_eval.OutputEval(args, str_data=data)
    evaluator.write_results()
