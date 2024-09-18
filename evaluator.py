import argparse
import subprocess
import yaml

# TODO: revert the edits I've currently made?
    # implement a standardized argparser
    # implement other standardized functions

# TODO: implement golds.yaml
    # currently goldretriever only works for fa and sr, the rest have a default gold URL in their script but fail to run
    # there is also no url for NER's evaluate.py, but we could just use the permalink in ner report
# TODO: reorganize file structure:
    # implement arguments.yaml now that I've made the file
    # create a single requirements.txt (add clams_utils), take advantage of golds.yaml too
    # have results files be produced in their specific eval directory
    # standardize output naming convention
    # simplify the directory structure as a whole to make it more user-friendly
# TODO: improve edgecase warning quality
    # currently accepts both directories and files, but an error is thrown if you assign a directory to an existing file, so potentially fix something here
        # solution to this will likely be standardizing the output
    # NER needs a warning that you can only use [-s [SOURCE_DIRECTORY]] [-o [OUT_DIRECTORY]] together, right now mixed up
        # consider handling this in NER itself, rather than within this script


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
        print(f"Standard Output:\n{e.stdout}")  # In case there's useful stdout before the error
        print(f"Error Output:\n{e.stderr}")  # The actual error message
    except FileNotFoundError:
        print(f"Script {script_name} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def construct_arguments(arg_dict, eval_type):
    """"Building the string argument from the input"""
    # argument to run in subprocess
    scripted_argument = []

    # get the dict of arguments from our yaml
    with open('arguments.yaml', 'r') as file:
        args_for_eval = yaml.safe_load(file)

    if eval_type in args_for_eval:
        for arg in args_for_eval[eval_type]:
            if args_for_eval[eval_type][arg] is None and arg_dict[arg] is not None:  # in cases where argument doesn't take a flag
                scripted_argument.extend([arg_dict[arg]])
            elif arg_dict[arg] is not None and arg_dict[arg] is not True and arg_dict[arg] is not False:  # all other valid argument cases
                scripted_argument.extend([args_for_eval[eval_type][arg], arg_dict[arg]])
            elif arg_dict[arg] is not None and arg_dict[arg] is True:  # in cases of boolean flags (i.e. --slate)
                scripted_argument.extend([args_for_eval[eval_type][arg]])
    else:
        raise ValueError(f"{eval_type} not a valid evaluation type, note that WIP evaluations are not supported")

    return scripted_argument


# this function could also include more information in the future
def specific_help(eval_type):
    """Prints the specific arguments that can be used for a certain eval when --specific-help is called"""

    # get the dict of arguments from our yaml
    with open('arguments.yaml', 'r') as file:
        args_for_eval = yaml.safe_load(file)

    # print the valid arguments
    if eval_type in args_for_eval:
        print("The following is the list of valid arguments (unordered) for this evaluation:")
        for argument in args_for_eval[eval_type]:
            print(argument)
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
    parser.add_argument('--side_by_side', nargs='?', help='directory to publish side-by-side results', default=None)
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


if __name__ == "__main__":
    main()
