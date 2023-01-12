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
        #if key not in string:
        #    return string

        result = ''
        buf = ''
        index = 0

        for i, ch in enumerate(string):
            if ch == key[index]:
                buf += ch
                index += 1
                if index == len(key):
                    if ((result == '' or result[-1] in self.separators) and
                            (i == len(string) - 1 or string[i + 1] in self.separators)):
                        result += self._abbrev_dict[key]
                    else:
                        result += buf
                    buf = ''
                    index = 0
            else:
                if index == 0:
                    result += ch
                else:
                    result += buf + ch
                    index = 0
                    buf = ''

        result += buf

        return result

    def set_dict(self, abbrev_dict: Dict[str, str]) -> None:
        self._abbrev_dict = abbrev_dict
        self._len_sorted_keys = sorted(list(abbrev_dict), key=len, reverse=True)

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        for key in self._len_sorted_keys:
            string = self._replace(string, key)

        return string
