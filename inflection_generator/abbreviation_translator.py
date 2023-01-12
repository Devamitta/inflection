from pathlib import Path
from typing import Dict

import pandas

from inflection_generator import settings


class AbbreviationTranslator:
    overrides_file = settings.DECLENSIONS_AND_CONJUGATIONS_OVERRIDES_FILE
    separators = " "
    iterations_threshold = 9999

    def __init__(self, script: str, declensions_file=settings.DECLENSIONS_AND_CONJUGATIONS_FILE):
        self._script = script
        abbrev_dict = self._dict_from_file(declensions_file)
        override_dict = self._dict_from_file(self.overrides_file)
        abbrev_dict.update(override_dict)

        self.set_dict(abbrev_dict)

    def _dict_from_file(self, file_path: Path) -> Dict[str, str]:
        abbrev_frame = pandas.read_excel(
            file_path,
            sheet_name="abbreviations",
            dtype=str,
            keep_default_na=True,)

        if self._script not in abbrev_frame:
            raise RuntimeError(f"No script variant {self._script} for abbreviations in {file_path}")

        # Filter rows with empty translation cell
        abbrev_frame = abbrev_frame[~abbrev_frame[self._script].isnull()]

        abbreviations = abbrev_frame["name"]
        translates = abbrev_frame[self._script]

        return dict(zip(abbreviations, translates))

    def _replace(self, string: str, key: str) -> str:
        counter = 0
        key_len = len(key)

        start = string.find(key)

        while start != -1:
            end = start + key_len
            if ((start == 0 or string[start - 1] in self.separators)
                    and (end == len(string) or string[end] in self.separators)):
                string = string.replace(key, self._abbrev_dict[key], 1)
            start = string.find(key, start)
            counter += 1
            if counter > self.iterations_threshold:
                raise RuntimeError('Too much cycles, seems like infinite loop')

        return string

    def set_dict(self, abbrev_dict: Dict[str, str]) -> None:
        self._abbrev_dict = abbrev_dict
        self._len_sorted_keys = sorted(list(abbrev_dict), key=len, reverse=True)

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        for key in self._len_sorted_keys:
            string = self._replace(string, key)

        print(f'== {string}')
        return string
