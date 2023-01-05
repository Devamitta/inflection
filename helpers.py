import os
import string

from datetime import datetime

import pandas

from pandas.errors import EmptyDataError

import settings


def create_directories() -> None:
    dirs = [
        settings.OUTPUT_DIR,
        settings.HTML_TABLES_DIR,
        "output/html suttas/",
        "output/inflections in table/",
        "output/inflections translit/",
        "output/inflections/",
        "output/patterns/",
        "output/pickle test/",
    ]

    for d in dirs:
        os.makedirs(str(d), exist_ok=True)


def data_frame_from_inflections_csv(file) -> pandas.DataFrame:
    try:
        result = pandas.read_csv(file, header=None, sep="\t")
    except (FileNotFoundError, EmptyDataError):
        result = pandas.DataFrame(data={0: [], 1: []})
    return result


def timeis():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"[blue]{current_time}[/blue]"


def excel_index(index: int) -> str:
    alphabet = list(string.ascii_uppercase)
    alph_len = len(alphabet)
    alphabet.insert(0, '0')

    result = ''
    correction = 1
    div = -1

    while div:
        div = index // (alph_len)
        rem = index % (alph_len) + correction
        index = div

        result = alphabet[rem] + result

        if len(result) == 1:
            correction = 0

    return result
