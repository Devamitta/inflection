import pandas

import settings


class AbbreviationTranslator:
    def __init__(self, script: str, declensions_file=settings.DECLENSIONS_AND_CONJUGATIONS_FILE):
        abbrev_frame = pandas.read_excel(
            declensions_file,
            sheet_name="abbreviations",
            dtype=str,
            keep_default_na=True,)

        if script not in abbrev_frame:
            raise RuntimeError(f'No script variant {script} for abbreviations in {declensions_file}')

        # Filter rows with empty trasnslate cell
        abbrev_frame = abbrev_frame[~abbrev_frame[script].isnull()]

        abbreviations = abbrev_frame['name']
        translates = abbrev_frame[script]

        self._abbrev_dict = dict(zip(abbreviations, translates))
        self._len_sorted_keys = sorted(abbreviations, key=len, reverse=True)

    def get(self, key: str, default=None) -> str:
        return self._abbrev_dict.get(key, default)

    def translate_string(self, string: str) -> str:
        for key in self._len_sorted_keys:
            val = self._abbrev_dict[key]
            string = string.replace(key, val)
        return string
