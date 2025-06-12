# from mos.arutils.args import define_data_args
from mos.utils.args import define_data_args

import os
import sys


def load_text_file():
    args = define_data_args()

    file_path = extract_file_path(args)
    max_lines = args.max_lines

    return TextLoader(file_path, max_lines)

def extract_file_path(args):
    data_root_dir = "./data/raw/"
    file_path = data_root_dir + args.train_file + ".txt"

    if not os.path.exists(file_path):
        print(f"Error: Training file '{file_path}' not found in '{data_root_dir}'", file=sys.stderr)
        sys.exit(1)

    return file_path

class TextLoader:
    def __init__(self, file_path, max_lines = 1000) -> None:
        self.file_path = file_path
        self.max_lines = max_lines
        self._load_contents_from_file()
        self.lines = self._get_lines()

    def _load_contents_from_file(self) -> None:
        with open(self.file_path, "r", encoding="UTF-8") as file:
            self.content = file.read()

    def _get_lines(self, delimiter = '\n') -> list[str]:
        return self.content.strip().split(delimiter)[:self.max_lines]