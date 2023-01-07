import os

from pathlib import Path

# TODO Better to use paths without spaces

# Source paths
DECLENSIONS_AND_CONJUGATIONS_FILE = Path("declensions & conjugations.xlsx")
DPS_DIR = Path(os.getenv("DPS_DIR", "../"))
CSCD_DIR = Path(os.getenv(
    "CSCD_DIR",
    "/home/deva/Documents/dpd-br/pure-machine-readable-corpus/cscd/"))  # TODO Path should not be absolute

# Output paths
OUTPUT_DIR = Path("output")
ALL_INFLECTIONS_FILE = OUTPUT_DIR / "all inflections.csv"
ALL_INFLECTIONS_TRANSLIT_FILE = OUTPUT_DIR / "all inflections translit.csv"
INFLECTIONS_DIR = OUTPUT_DIR / "inflections"
INFLECTIONS_TRANSLIT_DIR = OUTPUT_DIR / "inflections translit"
HTML_TABLES_DIR = OUTPUT_DIR / "html tables"
HTML_SUTTAS_DIR = OUTPUT_DIR / "html suttas"
