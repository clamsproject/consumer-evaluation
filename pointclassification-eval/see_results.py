import argparse
import base64
import csv
import os
from collections import defaultdict
from io import BytesIO
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json


def get_configs_and_macroavgs(directory):
    """
    1. Iterate over all files in the directory
    2. Get configuration information
    3. Retrieve the averages of precision, recall, and f1-score for each label for each configuration.
    4. Save and return them in a dictionary format.
    :param directory: where evaluation results files are stored
    :return: 1. A dictionary with ids as keys and the configuration dictionary as values : dict[id][parameter]->value
            2. A dictionary with ids as keys and the macro average dictionary as values: dict[id][label][metric]->value
    """
    result_sets = defaultdict(list)
    # Iterate over all files in the directory
    if directory == "":
        directory = os.getcwd()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # ignore subdirectories - there should only be .json files with config info and .csv files with results
        if os.path.isdir(file_path):
            continue
        result_sets[filename.split(".")[0]].append(file_path)

    # Store the evaluation results in the dictionary form
    macro_avgs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    configs = {}
    for key, value in result_sets.items():
        for file in value:
            if file.endswith(".csv"):
                csv_filename = os.path.basename(file).split(".")[0].split("@")[1]
                with open(file, "r") as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        label = row['labels'].split(" ")[0]
                        metric = row['labels'].split(" ")[1]
                        eval_type = row['labels'].split(" ")[2]
                        score = row['@@@ALL@@@']
                        if label in label_list:
                            macro_avgs[csv_filename][label][metric][eval_type] = score

            if file.endswith(".json"):
                with open(file, "r") as f:
                    data = json.load(f)
                del data['allowOverlap']
                configs[key] = data

    return configs, macro_avgs


def get_inverse_configs(configs):
    """
    Get inverse dictionary for configurations that allow user to find IDs from configurations.
    :param configs: A dictionary with IDs as keys and a dictionary with configurations as values.
    :return: A nested dictionary with parameter name as 1st keys, parameter value as 2nd key and a set of IDs as values.
    """
    inverse_dict = defaultdict(lambda: defaultdict(set))
    for key, val in configs.items():
        for k, v in val.items():
            inverse_dict[k][v].add(key)

    return inverse_dict


def get_grid(configs):
    """
    Get grid of configurations used.
    :param configs: A dictionary with IDs as keys and a dictionary with configurations as values.
    :return: A dictionary with parameter name as keys and list of parameter used in grid search as value.
    """
    grid = defaultdict(set)
    for value in configs.values():
        for k, v in value.items():
            grid[k].add(v)

    for key, val in grid.items():
        grid[key] = set(val)

    return grid


def get_pairs_to_compare(grid, inverse_configs, default_variable):
    """
    Get a list of pairs(lists of IDs) where all configurations are the same except for one given variable.
    :param grid: Grid of configurations used in this experiment.
    :param inverse_configs: A dictionary that allows user to search IDs from configurations.
    :param default_variable: The variable to be observed, which is 'minTPScore' for this app.
    :return: A list of pairs(lists of IDs)
    """

    # Delete variable key from grid and inverse_configs dictionary
    del grid[default_variable]
    del inverse_configs[default_variable]
    # Form all possible configurations of parameters from grid and store it as a list of dictionary form.
    conf_dicts = [dict(zip(grid.keys(), config)) for config in list(product(*grid.values()))]

    # Get all the possible lists of pairs(IDs) using inverse_configs dictionary and intersection of them for every configuration.
    pair_list = []
    for conf_dict in conf_dicts:
        list_of_sets = [inverse_configs[param_name][val] for param_name, val in conf_dict.items()]

        # Get intersection of sets of IDs for given configurations
        intersection_result = list_of_sets[0]
        # Iterate over the remaining sets and find the intersection
        for s in list_of_sets[1:]:
            intersection_result = intersection_result.intersection(s)

        pair_list.append(list(intersection_result))

    return pair_list


def compare_pairs(list_of_pairs, macroavgs, configs, default_variable, label_to_show, variable_values):
    """
    For list of pairs got from get_pairs_to_compare function, compare each pair by plotting bar graphs for given label.
    :param list_of_pairs: got from get_pairs_to_compare function for given variable
    :param macroavgs: A dictionary of macro averages of results retrieved from get_configs_and_macroavgs function.
    :param configs: A dictionary with IDs as keys and a dictionary with configurations as values.
    :param default_variable: The variable to be observed, which is 'minTPScore' for this app.
    :param label_to_show: User choice of label to show scores in the graph.
    """

    html = f'<html><head><title>Comparison of pairs: {label_to_show}</title></head><body>'

    # For each pair, retrieve corresponding data and plot a bar graph
    for pair in list_of_pairs:
        # re-order the pair to show the variable values in the same order as in the grid
        ordered_pair = [None] * len(variable_values)
        for i, value in enumerate(variable_values):
            for exp_id in pair:
                if configs[exp_id][default_variable] == value:
                    ordered_pair[i] = exp_id
        scores = macroavgs[ordered_pair[0]][label_to_show]
        data = defaultdict(lambda: defaultdict(dict))
        for i, exp_id in enumerate(ordered_pair):
            variable_value = configs[exp_id][default_variable]
            for metric, eval_types in scores.items():
                for eval_type in eval_types:
                    if label_to_show in macroavgs[exp_id]:
                        data[variable_value][metric][eval_type] = float(macroavgs[exp_id][label_to_show][metric][eval_type])
                    else:
                        data[variable_value][metric][eval_type] = 0.0

        # convert data to dataframe to load easier
        df = pd.DataFrame([[variable_value, metric, eval_type, score]
                           for variable_value, metrics in data.items()
                           for metric, eval_types in metrics.items()
                           for eval_type, score in eval_types.items()],
                          columns=['minTPScore', 'metric', 'Eval Type', 'Score'])

        fig = sns.catplot(data=df, x='metric', y='Score', hue='Eval Type', col='minTPScore', kind='bar', height=5, aspect=0.4)
        sns.move_legend(fig, loc='center right', fontsize='small', ncol=1, bbox_to_anchor=(1, 0.5))
        fig.tight_layout(pad=3)
        plt.subplots_adjust(top=0.85, bottom=0.2, left=0.06)
        plt.suptitle(str(label_to_show).title())

        for ax in fig.axes.flat:
            ax.set_xlabel(ax.get_title())
            ax.set_title('')
            ax.margins(x=0.1)
            ax.set_ylim(0.7, 1)
            for bar in ax.containers:
                ax.bar_label(bar, fmt='%.6s', fontsize='x-small', rotation='vertical', padding=3)

        # Show information on fixed parameters.
        string_configs = ""
        for k, v in configs[pair[0]].items():
            if k != 'minTPScore':
                string_configs += str(k) + ": " + str(v) + "\n"
        plt.text(2.7, 1, string_configs,
                verticalalignment='bottom', horizontalalignment='left',
                color='green', fontsize='small')

        temp_io_stream = BytesIO()
        fig.savefig(temp_io_stream, format='png', bbox_inches='tight')
        html += f'<p><img src="data:image/png;base64,{base64.b64encode(temp_io_stream.getvalue()).decode("utf-8")}"></p>'

        plt.cla()

    html += '</body></html>'
    with open(f'results-comparison-{default_variable}-{label_to_show}.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Directory with result and configuration files",
        default="",
    )
    parser.add_argument(
        '-l', '--label',
        default='bars',
        action='store',
        nargs='?',
        help='Pick a label to compare, default is bars'
    )

    args = parser.parse_args()

    # Get necessary dictionaries and lists for processing the comparison.
    label_list = ['bars', 'slate', 'chyron', 'credits']
    default_variable = 'minTPScore'
    configs, macroavgs = get_configs_and_macroavgs(args.directory)
    inverse_configs = get_inverse_configs(configs)
    grid = get_grid(configs)
    if args.label in label_list:
        choice_label = args.label
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument for label. Please enter one of {label_list}.")
    variable_values = sorted(grid[default_variable].copy())
    list_of_pairs = get_pairs_to_compare(grid.copy(), inverse_configs, default_variable)
    # Show the comparison results of pairs in bar graphs
    compare_pairs(list_of_pairs, macroavgs, configs.copy(), default_variable, choice_label, variable_values)
