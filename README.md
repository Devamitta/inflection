# Inflection list generator

## Description

Generate all inflections from scratch and write to CSV, HTML and text.

## Usage

> Run following commands from the root directory of the repo

Create and activate an env:

```shell
python3 -m venv env
source env/bin/activate
```

`source` command should be run for every new shell.

Install dependencies:

```shell
pip3 install -r requirements.txt
```

Directory with the dictionary sources may be set with `DPS_DIR` environment
variable:

```shell
export DPS_DIR='/PATH/TO/DIR/'
```

`DPS_DIR` directory expected to contain `spreadsheets` subdirectory with CSV
files.

Run generator with command:

```shell
python3 inflection_generator --kind DPS
```

Or in an old style:
```shell
python3 'inflection generator.py'
```

At first time utility should be ran mulitple times until output will not
changes.

## TODO

- Fix func args in `test*.py`
- Move majority of moudules to a subdir
