# Inflection list generator

This is an old version; the new version has been integrated into the DPD exporter:

https://github.com/digitalpalidictionary/dpd-db/

## Description

Generate all inflections from scratch and write to CSV, HTML and text.

## Usage

> Run following commands from the root directory of the repo

Create and activate a Python environment:

```shell
python3 -m venv env
source env/bin/activate
```

`source` command should be run for every new shell.

> Creating an environment is optional but recommended.

Install the package:

```shell
pip3 install -e .
```

> `-e` flag makes installations editable, i.e. package may be edited in place
> without reinstallation.

Directory with the dictionary sources may be set with `DPS_DIR` environment
variable:

```shell
export DPS_DIR='/PATH/TO/DIR/'
```

`DPS_DIR` directory expected to contain `spreadsheets` subdirectory with CSV
files.

Run generator with command:

```shell
inflection-generator --kind DPS
```

Or in an old style:
```shell
python3 'inflection generator.py'
```
