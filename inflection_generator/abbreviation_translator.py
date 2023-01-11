from typing import List

import pandas
import rich

from inflection_generator import settings
from inflection_generator.helpers import timeis


class AbbreviationTranslator:
    def __init__(self, script: str, declensions_file=settings.DECLENSIONS_AND_CONJUGATIONS_FILE):
        abbrev_frame = pandas.read_excel(
            declensions_file,
            sheet_name="abbreviations",
            dtype=str,
            keep_default_na=True,)

        if script not in abbrev_frame:
            raise RuntimeError(f"No script variant {script} for abbreviations in {declensions_file}")

        # Filter rows with empty translation cell
        abbrev_frame = abbrev_frame[~abbrev_frame[script].isnull()]

        abbreviations = abbrev_frame["name"]
        translates = abbrev_frame[script]

        self._abbrev_dict = dict(zip(abbreviations, translates))
        self._len_sorted_keys = sorted(abbreviations, key=len, reverse=True)

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        tokens = string.split()
        tokens_new: List[str] = []

        for tok in tokens:
            tok_new = self._abbrev_dict.get(tok)
            if tok_new is None:
                rich.print(f'{timeis()} [red] no translation for abbreviature "{tok}"')
                tok_new = tok
            tokens_new.append(tok_new)

        return " ".join(tokens_new)
