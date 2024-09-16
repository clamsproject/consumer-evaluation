import argparse
from pathlib import Path
import tempfile

from clams_utils.aapb import goldretriever, newshour_transcript_cleanup
from jiwer import wer
from mmif import Mmif, DocumentTypes

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils import standardized_parser

# constant:
## note that this repository is a private one and the files are not available to the public (due to IP concerns)
## hence using goldretriever to download the gold files WILL NOT work (goldretreiver is only for public repositories)
GOLD_URL = "https://github.com/clamsproject/aapb-collaboration/tree/89b8b123abbd4a9a67c525cc480173b52e0d05f0/21"


def get_text_from_mmif(mmif):
    with open(mmif, 'r') as f:
        mmif_str = f.read()
        data = Mmif(mmif_str)
        td_views = data.get_all_views_contain(DocumentTypes.TextDocument)
        if not td_views:
            for view in reversed(data.views):
                if view.has_error():
                    raise Exception("Error in the MMIF file: " + view.get_error().split('\n')[0])
                raise Exception("No TextDocument found in the MMIF file")
        annotation = next(td_views[-1].get_annotations(DocumentTypes.TextDocument))
        text = annotation.text_value

    return text


def get_text_from_txt(txt):
    with open(txt, 'r') as f:
        text = f.read()
    return text


# for now, we only care about casing, more processing steps might be added in the future
def process_text(text, ignore_case):
    if ignore_case:
        text = text.upper()
    return text


def calc_wer(hyp_file, gold_file, exact_case):
    # if we want to ignore casing
    hyp = process_text(get_text_from_mmif(hyp_file), not exact_case)
    gold = process_text(get_text_from_txt(gold_file), not exact_case)
    return wer(hyp, gold)


# check file id in preds and gold paths, and find the matching ids
def batch_run_wer(hyp_dir, gold_dir):
    hyp_dir = Path(hyp_dir)
    gold_dir = Path(gold_dir)
    
    hyp_files = hyp_dir.glob('*.mmif')
    gold_files = gold_dir.glob('*-transcript.txt')
    gold_files_dict = {x.stem.replace('-transcript', ''): x for x in gold_files}
    result = []

    for hyp_file in hyp_files:
        id_ = hyp_file.stem.split('.')[0]
        gold_file = gold_files_dict.get(id_)
        print("Processing file: ", hyp_file.name, gold_file.name if  gold_file else "(skip, no gold)")

        if gold_file:
            try:
                wer_result_exact_case = calc_wer(hyp_file, gold_file, True)
                wer_result_ignore_case = calc_wer(hyp_file, gold_file, False)
                result.append((id_, wer_result_exact_case, wer_result_ignore_case))
            except Exception as wer_exception:
                print("Error processing file: ", hyp_file.name, wer_exception)

    with open(f'results@{hyp_dir.name}.csv', 'w') as fp:
        fp.write('GUID,WER-case-sensitive,WER-case-insens\n')
        werS_sum = 0
        werI_sum = 0
        for r in result:
            fp.write(','.join(map(str, r)) + '\n')
            werS_sum += r[1]
            werI_sum += r[2]
        fp.write(f'Average,{werS_sum / len(result)},{werI_sum / len(result)}\n')


if __name__ == "__main__":
    # get the absolute path of video-file-dir and hypothesis-file-dir
    args = standardized_parser.parse_args()

    ref_dir = goldretriever.download_golds(GOLD_URL) if args.gold_file is None else args.gold_file
    audio_tmpdir = tempfile.TemporaryDirectory()
    newshour_transcript_cleanup.clean_and_write(ref_dir, audio_tmpdir.name)

    batch_run_wer(args.pred_file, audio_tmpdir.name)
