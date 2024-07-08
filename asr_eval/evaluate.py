import argparse
from clams_utils.aapb import goldretriever
from mmif import Mmif, Document, DocumentTypes
from jiwer import wer
import json
import os

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

def calculateWer(hyp_file, gold_file, exact_case):
    # if we want to ignore casing
    hyp = process_text(get_text_from_mmif(hyp_file), not exact_case)
    gold = process_text(get_text_from_txt(gold_file), not exact_case)
    return wer(hyp, gold)

# check file id in preds and gold paths, and find the matching ids
def batch_run_wer(hyp_dir, gold_dir): 
    hyp_files = os.listdir(hyp_dir)
    gold_files = os.listdir(gold_dir)
    gold_files_dict = {x.rsplit('-transcript.txt', 1)[0]: x for x in gold_files if x.endswith('-transcript.txt')}
    result = {}
    
    for hyp_file in hyp_files:
        id_ = hyp_file.split('.')[0]
        gold_file = gold_files_dict.get(id_)
        print("Processing file: ", hyp_file, gold_file, "PASS" if not gold_file else "EVAL")

        if gold_file:
            hyp_file_path = os.path.join(hyp_dir, hyp_file)
            gold_file_path = os.path.join(gold_dir, gold_file)
            try:
                wer_result_exact_case = calculateWer(hyp_file_path, gold_file_path, True)
                wer_result_ignore_case = calculateWer(hyp_file_path, gold_file_path, False)
                result[id_] = [
                        {
                            "wer_result": wer_result_exact_case,
                            "exact_case": True
                        },
                        {
                            "wer_result": wer_result_ignore_case,
                            "exact_case": False
                        }
                    ]
            except Exception as wer_exception:
                print("Error processing file: ", hyp_file, wer_exception)
    
    with open('results.json', 'w') as fp:
        fp.write(json.dumps(result, indent=2))


if __name__ == "__main__":
    # get the absolute path of video-file-dir and hypothesis-file-dir
    parser = argparse.ArgumentParser(description='Evaluate speech recognition results using WER.')
    parser.add_argument('-m', '--mmif-dir', type=str, required=True,
                        help='directory containing machine annotated files (MMIF)')
    parser.add_argument('-g', '--gold-dir', help='directory containing gold standard', default=None)
    args = parser.parse_args()

    ref_dir = goldretriever.download_golds(GOLD_URL) if args.gold_dir is None else args.gold_dir

    try: 
        batch_run_wer(args.mmif_dir, ref_dir)
    except Exception as batch_run_error:
        print(batch_run_error)
