import argparse
import subprocess

# TODO: implement golds.yaml
    # currently goldretriever only works for fa and sr, the rest have a default gold URL in their script but fail to run
    # there is also no url for NER's evaluate.py, but we could just use the permalink in ner report
# TODO: reorganize file structure:
    # create a single requirements.txt (add clams_utils), take advantage of golds.yaml too
    # make args_for_eval a yaml called yaml.help, implement it with specific-help
    # have results files be produced in their specific eval directory
    # standardize output naming convention
# TODO: update, delete, or move tests to a different file
    # remove tests or format them correctly for python tests, if keep, add more of the tests I had done before adding them here
# TODO: improve edgecase warning quality
    # make the script errors less ambiguous, find a way to read out what the errors are
    # currently accepts both directories and files, but an error is thrown if you assign a directory to an existing file, so potentially fix something here
    # NER needs a warning that you can only use [-s [SOURCE_DIRECTORY]] [-o [OUT_DIRECTORY]] together, right now mixed up


def run_script(script_name, constructed_arguments):
    """"Running the command script for the specific evaluation"""
    # build the script
    command = ['python', script_name] + constructed_arguments

    try:
        # run the script
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        print(e.output)
    except FileNotFoundError:
        print(f"Script {script_name} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def construct_arguments(arg_dict, eval_type):
    """"Building the string argument from the input"""
    scripted_argument = []  # argument to run in subprocess
    # below will be a yaml, but it is how we match each of our arguments to the subprocess arguments
    args_for_eval = {'asr_eval': {'pred_file': '-m', 'gold_file': '-g'},
                     'fa_eval': {'pred_file': '-m', 'gold_file': '-g', 'result_file': '-r', 'thresholds': '-t'},
                     'nel_eval': {'result_file': '-o', 'pred_file': None, 'gold_file': None},
                     'ner_eval': {'gold_file': '-g', 'pred_file': '-m', 'result_file': '-r', 'source_directory': '-s', 'side_by_side': '-o'},
                     'ocr_eval': {'pred_file': '-t', 'gold_file': '-g', 'result_file': '-o'},
                     'sr_eval': {'pred_file': '-m', 'gold_file': '-g', 'count_subtypes': '-s'},
                     'timeframe_eval': {'pred_file': '-m', 'side_by_side': '-s', 'result_file': '-r', 'gold_file': '-g', 'slate': '--slate', 'chyron': '--chyron'}
                     }

    if eval_type in args_for_eval:
        for arg in args_for_eval[eval_type]:
            if args_for_eval[eval_type][arg] is None and arg_dict[arg] is not None:  # in cases where argument doesn't take a flag
                scripted_argument.extend([arg_dict[arg]])
            elif arg_dict[arg] is not None and arg_dict[arg] is not True and arg_dict[arg] is not False:  # all other valid argument cases
                scripted_argument.extend([args_for_eval[eval_type][arg], arg_dict[arg]])
            elif arg_dict[arg] is not None and arg_dict[arg] is True:  # in cases of boolean flags (i.e. --slate)
                scripted_argument.extend([args_for_eval[eval_type][arg]])
            # handle any remaining args in the arg_dict that weren't used? currently the error is ambiguous
    else:
        raise ValueError(f"{eval_type} not a valid evaluation type, note that WIP evaluations are not supported")

    return scripted_argument


def specific_help(eval_type):
    """Prints the specific arguments that can be used for a certain eval when --specific-help is called"""
    # get this from gold.yaml once gold.yaml is updated
    # this could also include more information
    eval_dict = {
        "asr_eval": ["-m", "-g"],
        "fa_eval": ["-m", "-g", "[-r]", "[-t]"],
        "nel_eval": ["-m", "-g", "[-r]"],
        "ner_eval": ["-g", "-m", "[-r]", "[--side-by-side]", "[--source-directory]"],
        "ocr_eval": ["-m", "-g", "[-r]"],
        "sr_eval": ["-m", "-g", "[--count-subtypes]"],
        "timeframe_eval": ["-m",  "-g", "[-r]", "[--side-by-side]", "(--slate | --chyron)"]
    }

    if eval_type in eval_dict:
        print(" ".join(eval_dict[eval_type]))
    else:
        raise ValueError(f"{eval_type} not a valid evaluation type, note that WIP evaluations are not supported")


def main():
    parser = argparse.ArgumentParser(description='Run different types of evaluations.')
    parser.add_argument('-e', '--eval-type', type=str, required=True, help='Type of evaluation (e.g., eval1, eval2)')
    command_type = parser.add_mutually_exclusive_group(required=True)
    command_type.add_argument('--specific-help', action='store_true', help='Call -h for the given eval-type')
    command_type.add_argument('-m', '--pred-file', type=str, help='File containing predictions')
    parser.add_argument('-g', '--gold-file', type=str, help='File containing gold standard')
    timeframe_gold_group = parser.add_mutually_exclusive_group(required=False)
    timeframe_gold_group.add_argument('--slate', action='store_true', help='slate annotations')
    timeframe_gold_group.add_argument('--chyron', action='store_true', help='chyron annotations')
    parser.add_argument('--side-by-side', nargs='?', help='directory to publish side-by-side results', default=None)
    parser.add_argument('-f', '--result-file', nargs='?', help='file to store evaluation results', default='results.txt')
    # parser.add_argument('-o', '--output', nargs='?', help='path to print out eval result.', default=None)
    parser.add_argument('-t', '--thresholds',
                        help='comma-separated thresholds in seconds to count as "near-miss", use decimals ', type=str,
                        default="")
    parser.add_argument('--source-directory', nargs='?',
                        help="directory that contains original source files (without annotations)", default=None)
    parser.add_argument('--count-subtypes', action='store_true', default=False,
                        help='bool flag whether to consider subtypes for evaluation')

    args = parser.parse_args()
    # making sure eval_type is always the name of the directory
    if args.eval_type[-5:] == "_eval" or args.eval_type[-6:] == "_eval/":
        normalized_eval_type = args.eval_type
    else:
        normalized_eval_type = args.eval_type + "_eval"

    script_path = f"{normalized_eval_type.rstrip('/')}/evaluate.py"
    eval_type = normalized_eval_type.rstrip('/')

    # print(vars(args))  # run this for debugging or making tests
    if args.specific_help:
        specific_help(eval_type)
    else:
        arguments = construct_arguments(vars(args), eval_type)
        run_script(script_path, arguments)


# these will be more proper python tests in a future update, and will include more of the tests before pushing
def tests():
    # asr tests:
    print("testing asr...")
    script_path = "asr_eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'asr_eval/', 'specific_help': False,
                                     'pred_file': 'asr_eval/preds@whisper-wrapper-base@aapb-collaboration-21',
                                     'gold_file': 'golds/asr',
                                     'slate': False, 'chyron': False, 'side_by_side': None,
                                     'result_file': 'results.txt', 'thresholds': '', 'source_directory': None,
                                     'count_subtypes': False}, 'asr_eval')
    run_script(script_path, arguments)

    # fa tests:
    print("testing fa...")
    script_path = "fa_eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'fa_eval/', 'specific_help': False,
                                     'pred_file': 'fa_eval/preds@gentle-forced-aligner-wrapper@aapb-collaboration-21-nongoldtext/',
                                     'gold_file': '/Users/blambright/Downloads/clams/test-data/fa', 'slate': False,
                                     'chyron': False, 'side_by_side': None, 'result_file': 'results.txt',
                                     'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'fa_eval')
    run_script(script_path, arguments)

    # # ner tests:
    print("testing ner...")
    script_path = "ner-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'ner_eval/', 'specific_help': False,
                                     'pred_file': 'ner_eval/preds@spacy-wrapper@aapb-collaboration-21/',
                                     'gold_file': 'golds/ner/', 'slate': False, 'chyron': False, 'side_by_side': None,
                                     'result_file': 'results_ner', 'thresholds': '', 'source_directory': None,
                                     'count_subtypes': False}, 'ner_eval')
    run_script(script_path, arguments)

    # timeframe tests:
    print("testing timeframe...")
    script_path = "timeframe_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'timeframe_eval/', 'specific_help': False, 'pred_file': 'timeframe_eval/test-slate-preds/',
         'gold_file': 'golds/timeframe-slate-test/', 'slate': True, 'chyron': False, 'side_by_side': None,
         'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False},
        'timeframe_eval')
    run_script(script_path, arguments)

    # nel tests:
    print("testing nel...")
    script_path = "nel_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'nel_eval/', 'specific_help': False, 'pred_file': 'nel_eval/preds-test/',
         'gold_file': 'golds/nel-test/', 'slate': False, 'chyron': False, 'side_by_side': None,
         'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'nel_eval')
    run_script(script_path, arguments)

    # # ocr tests:
    print("testing ocr...")
    script_path = "ocr_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'ocr_eval/', 'specific_help': False, 'pred_file': 'ocr_eval/preds@tesseractocr-wrapper1.0@batch2',
         'gold_file': 'golds/ocr-batch2/', 'slate': False, 'chyron': False, 'side_by_side': None,
         'result_file': 'results_ocr', 'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'ocr_eval')
    run_script(script_path, arguments)

    # sr tests:
    print("testing sr...")
    script_path = "sr_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'sr_eval/', 'specific_help': False, 'pred_file': 'sr_eval/preds@app-swt-detection5.0@240117-aapb-collaboration-27-d', 'gold_file': 'golds/sr/', 'slate': False,
         'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '',
         'source_directory': None, 'count_subtypes': False}, 'sr_eval')
    run_script(script_path, arguments)


if __name__ == "__main__":
    # main()

    # tests
     tests()
