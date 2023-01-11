from pathlib import Path
from typing import Dict, List

import pandas
import rich

from inflection_generator import settings
from inflection_generator.helpers import timeis


class AbbreviationTranslator:
    overrides_file = settings.DECLENSIONS_AND_CONJUGATIONS_OVERRIDES_FILE

    def __init__(self, script: str, declensions_file=settings.DECLENSIONS_AND_CONJUGATIONS_FILE):
        abbrev_dict = self._dict_from_file(declensions_file, script)
        override_dict = self._dict_from_file(self.overrides_file, script)
        abbrev_dict.update(override_dict)

        self._abbrev_dict = abbrev_dict
        self._len_sorted_keys = sorted(list(abbrev_dict), key=len, reverse=True)

    @staticmethod
    def _dict_from_file(file_path: Path, script: str) -> Dict[str, str]:
        abbrev_frame = pandas.read_excel(
            file_path,
            sheet_name="abbreviations",
            dtype=str,
            keep_default_na=True,)

        if script not in abbrev_frame:
            raise RuntimeError(f"No script variant {script} for abbreviations in {file_path}")

        # Filter rows with empty translation cell
        abbrev_frame = abbrev_frame[~abbrev_frame[script].isnull()]

        abbreviations = abbrev_frame["name"]
        translates = abbrev_frame[script]

        return dict(zip(abbreviations, translates))

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        tokens = string.split()
        tokens_new: List[str] = []

        for tok in tokens:
            tok_new = self._abbrev_dict.get(tok)
            if tok_new is None:
                rich.print(f'{timeis()} [red] no translation for abbreviation "{tok}"')
                tok_new = tok
            tokens_new.append(tok_new)

        return " ".join(tokens_new)
