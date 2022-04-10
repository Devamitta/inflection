#!/usr/bin/env python3
# coding: utf-8 

from modules import *

def inflection_generator_for_dps():
	create_inflection_table_index()
	create_inflection_table_df()
	test_inflection_pattern_changed()
	create_dps_df()
	test_for_missing_stem_and_pattern()
	test_for_wrong_patterns()
	test_for_differences_in_stem_and_pattern()
	test_if_inflections_exist_dps()
	generate_changed_inflected_forms()
	generate_html_inflection_table()
	transcribe_new_inflections()
	combine_old_and_new_translit_dataframes()
	export_translit_to_pickle()

if __name__ == "__main__":
   inflection_generator_for_dps()