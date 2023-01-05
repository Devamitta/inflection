import os

from pathlib import Path

# Source paths
DECLENSIONS_AND_CONJUGATIONS_FILE = Path("declensions & conjugations.xlsx")
DPS_DIR = Path(os.getenv("DPS_DIR", "../spreadsheets/"))

# Output paths
OUTPUT_DIR = Path("output")
ALL_INFLECTIONS_FILE = OUTPUT_DIR / "all inflections.csv"
ALL_INFLECTIONS_TRANSLIT_FILE = OUTPUT_DIR / "all inflections translit.csv"
HTML_TABLES_DIR = OUTPUT_DIR / "html tables"
