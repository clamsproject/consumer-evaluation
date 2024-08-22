import argparse
import subprocess

# TODO: create a single requirements.txt (add clams_utils), take advantage of golds.yaml too, make args_for_eval a yaml
# TODO: implement a -h for each one, make --specific-help/-e and slate/chyron an either/or situation
# TODO: NER needs a warning that you can only use [-s [SOURCE_DIRECTORY]] [-o [OUT_DIRECTORY]] together, right now mixed up
# TODO: change specific help to show the future arguments yaml-m
# TODO: have results files be produced in their specific eval directory
# TODO: refactor slate/chyron and count-subtypes to be consistent in how they take arguments
# TODO: remove tests or format them correctly for python tests, if keep, add more of the tests I had done before adding them here
# TODO: make the script errors less ambiguous?
# TODO: currently accepts both directories and files, but an error is thrown if you assign a directory to an existing file, so potentially fix something here
# TODO: consider making the structure of this script more like a module, and refactor the other evaluation.py, so that it can be used for future evaluation scripts as wellc
# TODO: standardize output

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


def construct_arguments(arg_dict):
    """"Building the string argument from the input"""
    scripted_argument = []  # argument to run in subprocess
    # below will be a yaml, but it is how we match each of our arguments to the subprocess arguments
    args_for_eval = {'asr_eval': {'pred_file': '-m', 'gold_file': '-g'},
                     'fa_eval': {'pred_file': '-m', 'gold_file': '-g', 'result_file': '-r', 'thresholds': '-t'},
                     'nel_eval': {'result_file': '-o', 'pred_file': None, 'gold_file': None},
                     'ner_eval': {'gold_file': '-g', 'pred_file': '-m', 'result_file': '-r', 'source_directory': '-s', 'side_by_side': '-o'},
                     'ocr_eval': {'pred_file': '-t', 'gold_file': '-g', 'result_file': '-o'},
                     'sr-eval': {'pred_file': '-m', 'gold_file': '-g', 'count_subtypes': '-s'},
                     'timeframe-eval': {'pred_file': '-m', 'side_by_side': '-s', 'result_file': '-r', 'gold_file': '-g', 'slate': '--slate', 'chyron': '--chyron'}
                     }

    eval_type = arg_dict['eval_type'].strip('/')
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


def main():
    parser = argparse.ArgumentParser(description='Run different types of evaluations.')
    parser.add_argument('-e', '--eval-type', type=str, required=True, help='Type of evaluation (e.g., eval1, eval2)')
    parser.add_argument('--specific-help', action='store_true', help='Call -h for the given eval-type')
    parser.add_argument('-m', '--pred-file', type=str, help='File containing predictions')
    parser.add_argument('-g', '--gold-file', type=str, help='File containing gold standard')
    parser.add_argument('--slate', action='store_true', help='slate annotations')
    parser.add_argument('--chyron', action='store_true', help='chyron annotations')
    parser.add_argument('--side-by-side', nargs='?', help='directory to publish side-by-side results', default=None)
    parser.add_argument('-r', '--result-file', nargs='?', help='file to store evaluation results', default='results.txt')
    parser.add_argument('-t', '--thresholds',
                        help='comma-separated thresholds in seconds to count as "near-miss", use decimals ', type=str,
                        default="")
    parser.add_argument('--source-directory', nargs='?',
                        help="directory that contains original source files (without annotations)", default=None)
    # parser.add_argument('-o', '--output', nargs='?', help='path to print out eval result.', default=None)
    parser.add_argument('--count-subtypes', action='store_true', default=False,
                        help='bool flag whether to consider subtypes for evaluation')

    args = parser.parse_args()

    print(vars(args))  # run this for debugging or making tests
    script_path = f"{args.eval_type.strip('/')}/evaluate.py"  # make this more flexible later
    if args.specific_help:
        run_script(script_path, ['-h'])
    else:
        arguments = construct_arguments(vars(args))
        run_script(script_path, arguments)


# these will be more proper python tests in a future update, and will include more of the tests before pushing
def tests():
    # asr tests:
    script_path = "asr-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'asr_eval/', 'specific_help': False, 'pred_file': 'asr_eval/preds@whisper-wrapper-base@aapb-collaboration-21', 'gold_file': '/Users/blambright/Downloads/clams/aapb-evaluations-main/asr_eval/data/21', 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # fa tests:
    script_path = "fa-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'fa_eval/', 'specific_help': False, 'pred_file': 'fa_eval/preds@gentle-forced-aligner-wrapper@aapb-collaboration-21-nongoldtext/', 'gold_file': '/Users/blambright/Downloads/clams/test-data/fa', 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # ner tests:
    script_path = "ner-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'ner_eval/', 'specific_help': False, 'pred_file': 'ner_eval/preds@spacy-wrapper@aapb-collaboration-21/', 'gold_file': 'golds/ner/', 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # timeframe tests
    script_path = "timeframe-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'timeframe-eval/', 'specific_help': False, 'pred_file': 'timeframe-eval/test-slate-preds/', 'gold_file': 'golds/timeframe-slate-test/', 'slate': True, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # nel tests
    script_path = "nel-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'nel_eval/', 'specific_help': False, 'pred_file': 'nel_eval/preds-test/', 'gold_file': 'golds/nel-test/', 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # ocr tests
    script_path = "ocr-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'ocr_eval/', 'specific_help': False, 'pred_file': 'ocr_eval/preds@tesseractocr-wrapper1.0@batch2', 'gold_file': 'golds/ocr-batch2/', 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results_dir', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)

    # sr tests
    script_path = "sr-eval/evaluate.py"
    # test 1
    arguments = construct_arguments({'eval_type': 'sr-eval/', 'specific_help': False, 'pred_file': 'golds/sr/', 'gold_file': None, 'slate': False, 'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False})
    run_script(script_path, arguments)


if __name__ == "__main__":
    main()

    # tests
    # tests()
