import argparse
from collections import defaultdict, Counter
import pathlib
import pandas as pd
import json
from clams_utils.aapb import goldretriever

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval_utils import standardized_parser, basic_eval

# constant:
GOLD_URL = "https://github.com/clamsproject/aapb-annotations/tree/bebd93af0882b8cf942ba827917938b49570d6d9/scene-recognition/golds"
# note that you must first have output mmif files to compare against

# parse SWT output into dictionary to extract label-timepoint pairs

# convert ISO timestamp strings (hours:minutes:seconds.ms) back to milliseconds


def convert_iso_milliseconds(timestamp):
    ms = 0
    # add hours
    ms += int(timestamp.split(":")[0]) * 3600000
    # add minutes
    ms += int(timestamp.split(":")[1]) * 60000
    # add seconds and milliseconds
    ms += float(timestamp.split(":")[2]) * 1000
    ms = int(ms)
    return ms

# extract gold pairs from each csv. note goldpath is fed in as a path object
def extract_gold_labels(goldpath, count_subtypes=False):
    df = pd.read_csv(goldpath)
    # convert timestamps (iso) back to ms
    df['timestamp'] = df['timestamp'].apply(convert_iso_milliseconds)
    if count_subtypes:
        # fill empty subtype rows with '' then concatenate with type label
        df['subtype label'] = df['subtype label'].fillna("")
        df['combined'] = df['type label'] + ":" + df['subtype label']
        # trim extra ":"
        df['combined'] = df['combined'].apply(lambda row: row[:-1] if row[-1] == ':' else row)
        # create dictionary of 'timestamp':'combined' from dataframe
        gold_dict = df.set_index('timestamp')['combined'].to_dict()
    else:
        # ignore subtype label column
        gold_dict = df.set_index('timestamp')['type label'].to_dict()
    # return dictionary that maps timestamps to label
    return gold_dict

# method to match a given predicted timestamp (key) with the closest gold timestamp:
# acceptable range is default +/- 5 ms. if nothing matches, return None

def closest_gold_timestamp(pred_stamp, gold_dict, good_range = 5):
    # first check if pred in gold_dict. if yes, return pred
    if pred_stamp in gold_dict:
        return pred_stamp
    # for i = 5 to 1 check if pred - i in gold_dict, if yes return pred - i
    for i in range(good_range, 0, -1):
        if pred_stamp - i in gold_dict:
            return pred_stamp - i
    # for i = 1 to i = 5 check if pred + i in gold dict, if yes return pred + i
    for i in range(1, good_range + 1):
        if pred_stamp + i in gold_dict:
            return pred_stamp + i
    return None

# extract predicted label pairs from output mmif and match with gold pairs
# note that pred_path is already a filepath, not a string
# returns a dictionary with timestamps as keys and tuples of labels as values.


def extract_predicted_consolidate(pred_path, gold_dict, count_subtypes = False):
    # create a dictionary to fill in with timestamps -> label tuples (predicted, gold)
    combined_dict = {}
    with open(pred_path, "r") as file:
        pred_json = json.load(file)
        for view in pred_json["views"]:
            if "annotations" in view:
                for annotation in view["annotations"]:
                    if "timePoint" in annotation['properties']:
                        # match pred timestamp to closest gold timestamp
                        # using default range (+/- 5ms)
                        curr_timestamp = closest_gold_timestamp(annotation['properties']['timePoint'], gold_dict)
                        # check if closest_gold_timestamp returned None (not within acceptable range)
                        if not curr_timestamp:
                            continue
                        # truncate label if count_subtypes is false
                        pred_label = annotation['properties']['label'] if count_subtypes else annotation['properties']['label'][0]
                        # if NEG set to '-'
                        if annotation['properties']['label'] == 'NEG':
                            pred_label = '-'
                        # put gold and pred labels into combined dictionary
                        combined_dict[curr_timestamp] = (pred_label, gold_dict[curr_timestamp])
    return combined_dict

# calculate document-level p, r, f1 for each label and macro avg. also returns total counts
# of tp, fp, fn for each label to calculate micro avg later.
def document_evaluation(combined_dict):
    # count up tp, fp, fn for each label
    total_counts = defaultdict(Counter)
    for timestamp in combined_dict:
        pred, gold = combined_dict[timestamp][0], combined_dict[timestamp][1]
        if pred == gold:
            total_counts[pred]["tp"] += 1
        else:
            total_counts[pred]["fp"] += 1
            total_counts[gold]["fn"] += 1
    # calculate P, R, F1 for each label, store in nested dictionary
    scores_by_label = defaultdict(lambda: defaultdict(float))
    # running total for (macro) averaged scores per document
    average_p = 0
    average_r = 0
    average_f1 = 0
    # counter to account for unseen labels
    unseen = 0
    for label in total_counts:
        tp, fp, fn = total_counts[label]["tp"], total_counts[label]["fp"], total_counts[label]["fn"]
        # if no instances are present/predicted, account for this when taking average of scores
        if tp + fp + fn == 0:
            unseen += 1
        precision = float(tp/(tp + fp)) if (tp + fp) > 0 else 0
        recall = float(tp/(tp + fn)) if (tp + fn) > 0 else 0
        f1 = float(2*(precision*recall)/(precision + recall)) if (precision + recall) > 0 else 0
        # add individual scores to dict and then add to running sum
        scores_by_label[label]["precision"] = precision
        scores_by_label[label]["recall"] = recall
        scores_by_label[label]["f1"] = f1
        average_p += precision
        average_r += recall
        average_f1 += f1
    # calculate macro averages for document and add to scores_by_label
    # make sure to account for unseen unpredicted labels
    denominator = len(scores_by_label) - unseen
    scores_by_label["average"]["precision"] = float(average_p / denominator)
    scores_by_label["average"]["recall"] = float(average_r / denominator)
    scores_by_label["average"]["f1"] = float(average_f1 / denominator)
    # return both scores_by_label and total_counts (to calculate micro avg later)
    return scores_by_label, total_counts

# once you have processed every document, this method runs to calculate the micro-averaged
# scores. the input is a list of total_counts dictionaries, each obtained from running
# document_evaluation.
def total_evaluation(total_counts_list):
    # create dict to hold total tp, fp, fn for all labels
    total_instances_by_label = defaultdict(Counter)
    # iterate through total_counts_list to get complete count of tp, fp, fn by label
    for doc_dict in total_counts_list:
        for label in doc_dict:
            total_instances_by_label[label]["tp"] += doc_dict[label]["tp"]
            total_instances_by_label[label]["fp"] += doc_dict[label]["fp"]
            total_instances_by_label[label]["fn"] += doc_dict[label]["fn"]
            # include a section for total tp/fp/fn for all labels
            total_instances_by_label["all"]["tp"] += doc_dict[label]["tp"]
            total_instances_by_label["all"]["fp"] += doc_dict[label]["fp"]
            total_instances_by_label["all"]["fn"] += doc_dict[label]["fn"]
    # create complete_micro_scores to store micro avg scores for entire dataset
    complete_micro_scores = defaultdict(lambda: defaultdict(float))
    # fill in micro scores
    for label in total_instances_by_label:
        tp, fp, fn = (total_instances_by_label[label]["tp"], total_instances_by_label[label]["fp"],
                      total_instances_by_label[label]["fn"])
        precision = float(tp/(tp + fp)) if (tp + fp) > 0 else 0
        recall = float(tp/ (tp + fn)) if (tp + fn) > 0 else 0
        f1 = float(2*precision*recall/(precision + recall)) if (precision + recall) > 0 else 0
        complete_micro_scores[label]["precision"] = precision
        complete_micro_scores[label]["recall"] = recall
        complete_micro_scores[label]["f1"] = f1
    return complete_micro_scores

# run the evaluation on each predicted-gold pair of files, and then the entire dataset for
# micro average
def run_dataset_eval(mmif_dir, gold_dir, count_subtypes):
    # create dict of guid -> scores to store each dict of document-level scores
    doc_scores = {}
    # create list to store each dict of document-level counts
    document_counts = []
    mmif_files = pathlib.Path(mmif_dir).glob("*.mmif")
    # get each mmif file
    for mmif_file in mmif_files:
        guid = ""
        with open(mmif_file, "r") as f:
            curr_mmif = json.load(f)
            # get guid
            location = curr_mmif["documents"][0]["properties"]["location"]
            guid = location.split("/")[-1].split(".")[0]
        # match guid with gold file
        gold_file = next(pathlib.Path(gold_dir).glob(f"*{guid}*"))
        # process gold
        gold_dict = extract_gold_labels(gold_file, count_subtypes)
        # process predicted and consolidate
        combined_dict = extract_predicted_consolidate(mmif_file, gold_dict, count_subtypes)
        # evaluate on document level, storing scores in document_scores and counts in document_counts
        eval_result = document_evaluation(combined_dict)
        doc_scores[guid] = eval_result[0]
        document_counts.append(eval_result[1])
    # now after processing each document and storing the relevant scores, we can evaluate the
    # dataset performance as a whole
    data_scores = total_evaluation(document_counts)
    return doc_scores, data_scores

def separate_score_outputs(doc_scores, dataset_scores, mmif_dir):
    # get name for new directory
    # with our standard, this results in "scores@" appended to the batch name
    batch_score_name = "scores@" + mmif_dir.split('@')[-1].strip('/')
    # create new dir for scores based on batch name
    new_dir = pathlib.Path.cwd() / batch_score_name
    new_dir.mkdir(parents = True, exist_ok = True)
    # iterate through nested dict, output separate scores for each guid
    # for guid in doc_scores:
    #     doc_df = pd.DataFrame(doc_scores[guid])
    #     doc_df = doc_df.transpose()
    #     out_path = new_dir / f"{guid}.csv"
    #     doc_df.to_csv(out_path)
    # output total dataset scores
    # dataset_df = pd.DataFrame(dataset_scores)
    # dataset_df = dataset_df.transpose()
    # dataset_df.to_csv(new_dir/"dataset_scores.csv")
    doc_scores["Average"] = dataset_scores
    return doc_scores


if __name__ == "__main__":
    args = standardized_parser.parse_args()
    mmif_dir = args.pred_file
    gold_dir = goldretriever.download_golds(GOLD_URL) if args.gold_file is None else args.gold_file
    count_subtypes = args.count_subtypes
    document_scores, dataset_scores = run_dataset_eval(mmif_dir, gold_dir, count_subtypes)
    # document scores are for each doc, dataset scores are for overall (micro avg)
    # call method to output scores for each doc and then for total scores
    data = separate_score_outputs(document_scores, dataset_scores, mmif_dir)

    evaluator = basic_eval.OutputEval(args, dict_data=data)
    evaluator.write_results()
