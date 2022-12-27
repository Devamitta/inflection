import os
import sys
import warnings

from modules import *

warnings.simplefilter(action='ignore', category=FutureWarning)

# class_file_name = sys.argv[1]

def inflection_exists_in_sutta():
    class_file_name = sys.argv[1]

    # convert_dpd_ods_to_csv()
    create_inflection_table_index()
    create_inflection_table_df()
    test_inflection_pattern_changed()
    create_sbs_df(class_file_name)
    test_for_missing_stem_and_pattern()
    test_for_wrong_patterns()
    test_for_differences_in_stem_and_pattern()
    test_if_inflections_exist_suttas()
    generate_changed_inflected_forms()
    combine_old_and_new_dataframes()
    export_inflections_to_pickle()
    make_list_of_all_inflections()
    make_list_of_all_inflections_no_meaning()
    # higlight
    make_list_of_all_inflections_only_in_class()
    # red
    make_list_of_all_inflections_already_in()
    # green
    make_list_of_all_inflections_potential(class_file_name)
    # blue
    read_and_clean_sutta_text()
    make_comparison_table()
    html_find_and_replace()
    write_html()
    # open_in_browser()

inflection_exists_in_sutta()
