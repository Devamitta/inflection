from pathlib import Path
from typing import Dict

import pandas

from inflection_generator import settings


class AbbreviationTranslator:
    # TODO Docstring
    overrides_file = settings.DECLENSIONS_AND_CONJUGATIONS_OVERRIDES_FILE
    separators = " "

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
        if key not in string:
            return string

        result = ''
        buf = ''
        key_ind = 0

        for s_ind, ch in enumerate(string):
            if ch == key[key_ind]:
                buf += ch
                key_ind += 1
                if key_ind == len(key):
                    # Check if token surrounded with whitespaces or bounds
                    if ((result == '' or result[-1] in self.separators) and
                            (s_ind == len(string) - 1 or string[s_ind + 1] in self.separators)):
                        result += self._abbrev_dict[key]
                    else:
                        result += buf
                    buf = ''
                    key_ind = 0
            else:
                if key_ind == 0:
                    result += ch
                else:
                    result += buf + ch
                    key_ind = 0
                    buf = ''

        result += buf

        return result

    def set_dict(self, abbrev_dict: Dict[str, str]) -> None:
        self._abbrev_dict = abbrev_dict
        self._len_sorted_keys = sorted(list(abbrev_dict), key=len, reverse=True)

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        with open('str.txt', 'a') as f:  # FIXME
            f.write(string + '\n')
        for key in self._len_sorted_keys:
            string = self._replace(string, key)

        return string
