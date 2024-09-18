import json

# TODO: implement a side-by-side output function?
# TODO: implement a function for common evaluation metrics?
# TODO: see if we can have a common mmif extraction function
# TODO: see if we can have a common gold data extraction function
# TODO: see if we can standardize the use of goldretriever and the other stuff in the main() of eval scripts
# TODO: see if we can match golds to preds in a standard way? is it worth it?
# TODO: standardize a get guid function?


class Eval:
    def __init__(self, arguments, dict_data=None, str_data=None):
        self.arguments = arguments
        self.dict_data = dict_data  # refactored to be in the form of a str
        self.str_data = str_data

    def write_results(self):
        """Write evaluation results to txt file."""
        # write to a txt file, we'll do this for every eval
        try:
            with open(self.arguments.result_file, 'w') as fh_out:
                if self.dict_data:
                    json.dump(self.dict_data, fh_out, indent=4)
                    fh_out.write("\n")
                if self.str_data:
                    fh_out.write(self.str_data)
        except AttributeError as e:
            print(f"Error result_file not in args: {e}")

    # TODO: implement this function
    # TODO: implentation of this is different when extracting timeframes
    # from asr
    # see if the nel is comparable, and ner
    # ocr has something slightly similar
    # timeframe eval has a nice one too that is timeframe level
    def get_text_from_mmif(self, mmif):
        """Getting data from a mmif file"""
        ...

    # from asr
    # TODO: implement this function, potentially just combine it with get_text_from_mmif
    # ner has something similar
    def get_text_from_txt(self, txt):
        """Getting the data from a text file"""
        ...

    # TODO: see if this is worth it, or we would only use it for the timeframes
    # from fa
    # see if the nel is comparable
    # ocr has a kind of csv version of this
    # sr has a version of this, it's basically extracting data from gold labels, but this is the case for csv and tsv
    # In general, there are a lot of things related to extracting timestamps in general, maybe generalize that?
    # timeframes and sr have something kinda similar too
    def read_cadet_annotation_tsv(self, tsv_file_list):
        """Converts tsv to timeframes"""
        ...

    # TODO: see if we can implement this for every file, I feel like every one has a version
    # from nel
    # ner has something similar
    def match_files(self, test_dir, gold_dir) -> list:
        """Compare the files in the gold and test directories. Return pairs of matching files in a list.
        :param test_dir: Directory of test .mmif files
        :param gold_dir: Directory of gold .tsv files
        :return: list of tuples containing corresponding data file locations in (test, gold) format.
        """
        ...

    # TODO: standardize these should be in all of them
    # from ner, ocr
    def get_guid(triple):
        """returns guid for a triple of files"""
        ...

    # TODO: see if this is useful for standardization, could include identifying other file types too
    # from nel
    # ner has something very similar
    def file_to_ne(file_path: str) -> list:
        """Checks whether the file is in .mmif or .tsv format and calls the appropriate function
        to get a list of NEL objects"""
        ...
