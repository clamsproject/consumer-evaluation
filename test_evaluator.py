from evaluator import construct_arguments, run_script

# TODO: change the output directory to always be in a tests folder, and put this in the test folder too
# TODO: potentially implement these as assertions so we don't have to check the outputs manually?
# TODO: make sure none of my own absolute paths are in this

def asr_test():
    print("testing asr...")
    script_path = "asr_eval/evaluate.py"
    # test 1
    arguments = construct_arguments({
                                     'pred_file': 'asr_eval/preds@whisper-wrapper-base@aapb-collaboration-21',
                                     'gold_file': 'golds/asr',
                                     'slate': False, 'chyron': False, 'side_by_side': None,
                                     'result_file': 'results.txt', 'thresholds': '', 'source_directory': None,
                                     'count_subtypes': False}, 'asr_eval')
    run_script(script_path, arguments)

def fa_test():
    print("testing fa...")
    script_path = "fa_eval/evaluate.py"
    # test 1
    arguments = construct_arguments({
                                     'pred_file': 'fa_eval/preds@gentle-forced-aligner-wrapper@aapb-collaboration-21-nongoldtext/',
                                     'gold_file': '/Users/blambright/Downloads/clams/test-data/fa', 'slate': False,
                                     'chyron': False, 'side_by_side': None, 'result_file': 'results.txt',
                                     'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'fa_eval')
    run_script(script_path, arguments)

def ner_test():
    print("testing ner...")
    script_path = "ner_eval/evaluate.py"
    # test 1
    arguments = construct_arguments({
                                     'pred_file': 'ner_eval/preds@spacy-wrapper@aapb-collaboration-21/',
                                     'gold_file': 'golds/ner/', 'slate': False, 'chyron': False,
                                     'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '',
                                     'source_directory': None,
                                     'count_subtypes': False}, 'ner_eval')
    run_script(script_path, arguments)

def timeframe_test():
    print("testing timeframe...")
    script_path = "timeframe_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'timeframe_eval/', 'specific_help': False, 'pred_file': 'timeframe_eval/test-slate-preds/',
         'gold_file': 'golds/timeframe-slate-test/', 'slate': True, 'chyron': False, 'side_by_side': None,
         'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False},
        'timeframe_eval')
    run_script(script_path, arguments)

def nel_test():
    print("testing nel...")
    script_path = "nel_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'specific_help': False, 'pred_file': 'nel_eval/preds-test/',
         'gold_file': 'golds/nel-test/', 'slate': False, 'chyron': False, 'side_by_side': None,
         'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'nel_eval')
    run_script(script_path, arguments)

def ocr_test():
    print("testing ocr...")
    script_path = "ocr_eval/evaluate.py"
    # test 1
    # note that this is only replicable with preds@parseqocr-wrapper1.0@batch2
    arguments = construct_arguments(
        {'pred_file': 'ocr_eval/preds@parseqocr-wrapper1.0@batch2',
         'gold_file': 'golds/ocr-batch2/', 'slate': False, 'chyron': False, 'side_by_side': None,
         'result_file': 'results.txt', 'thresholds': '', 'source_directory': None, 'count_subtypes': False}, 'ocr_eval')
    print(arguments)
    run_script(script_path, arguments)

def sr_test():
    print("testing sr...")
    script_path = "sr_eval/evaluate.py"
    # test 1
    arguments = construct_arguments(
        {'eval_type': 'sr_eval/', 'specific_help': False, 'pred_file': 'sr_eval/preds@app-swt-detection5.0@240117-aapb-collaboration-27-d', 'gold_file': 'golds/sr/', 'slate': False,
         'chyron': False, 'side_by_side': None, 'result_file': 'results.txt', 'thresholds': '',
         'source_directory': None, 'count_subtypes': False}, 'sr_eval')
    run_script(script_path, arguments)

def all_tests():
    asr_test()
    fa_test()
    ner_test()
    timeframe_test()
    nel_test()
    ocr_test()
    sr_test()


sr_test()