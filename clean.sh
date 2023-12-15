#!/usr/bin/env bash

cd output

# remove all files from dir and subdirs exept *.csv
find . -type f ! -name '*.csv' -delete

# replace all *.csv files from dir with empty once with the same names
for file in *.csv; do echo -n > "$file"; done

