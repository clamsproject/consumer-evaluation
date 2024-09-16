import argparse
import yaml

# TODO: potentially make the parsing of directories more flexible using regex


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
            elif arg_dict[arg] is not None and arg_dict[arg] is not True and arg_dict[
                arg] is not False:  # all other valid argument cases
                scripted_argument.extend([args_for_eval[eval_type][arg], arg_dict[arg]])
            elif arg_dict[arg] is not None and arg_dict[arg] is True:  # in cases of boolean flags (i.e. --slate)
                scripted_argument.extend([args_for_eval[eval_type][arg]])
    else:
        raise ValueError(f"{eval_type} not a valid evaluation type, note that WIP evaluations are not supported")

    return scripted_argument


def parse_args():
    """Standardized argparse, ensures that every one of our evals has the same structure for parsing arguments"""
    parser = argparse.ArgumentParser(description='Run different types of evaluations.')
    parser.add_argument('-m', '--pred-file', type=str, help='File containing predictions')
    parser.add_argument('-g', '--gold-file', type=str, help='File containing gold standard')
    timeframe_gold_group = parser.add_mutually_exclusive_group(required=False)
    timeframe_gold_group.add_argument('--slate', action='store_true', help='slate annotations')
    timeframe_gold_group.add_argument('--chyron', action='store_true', help='chyron annotations')
    parser.add_argument('--side-by-side', nargs='?', help='directory to publish side-by-side results', default=None)
    parser.add_argument('-r', '--result-file', nargs='?', help='file to store evaluation results',
                        default='results.txt')
    parser.add_argument('-t', '--thresholds',
                        help='comma-separated thresholds in seconds to count as "near-miss", use decimals ',
                        type=str,
                        default="")
    parser.add_argument('--source-directory', nargs='?',
                        help="directory that contains original source files (without annotations)", default=None)
    parser.add_argument('--count-subtypes', action='store_true', default=False,
                        help='bool flag whether to consider subtypes for evaluation')

    args = parser.parse_args()

    return args
    # script_path = f"{args.eval_type.rstrip('/')}/evaluate.py"
    # eval_type = args.eval_type.rstrip('/')
    #
    # # print(vars(args))  # run this for debugging or making tests
    # arguments = self.construct_arguments(vars(args), eval_type)
    # return arguments, script_path
