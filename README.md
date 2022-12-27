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

Install dependencies:

```shell
pip3 install -r requirements.txt
```

Directory with CSV files of dictionary may be set with `DPS_DIR` environment
variable:

```shell
export DPS_DIR='/PATH/TO/FILES/'
```

Run generator with command:

```shell
python 'inflection generator.py'
```

At first time utility should be ran mulitple times until output will not
changes.

## TODO

- Merge generator modules
