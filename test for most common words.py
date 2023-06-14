# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

import sys
import warnings

from inflection_generator import modules, settings

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    class_file_name = sys.argv[1]

    # modules.convert_dpd_ods_to_csv()
    inflection_table_index = modules.create_inflection_table_index()
    inflection_table = modules.create_inflection_table_df()
    modules.test_inflection_pattern_changed(inflection_table_index, inflection_table)

    csv_file = settings.DPS_DIR/"word-frequency"/"csv-for-examples"/f"{class_file_name}-class.csv"
    data, _ = modules.create_sbs_df(csv_file)

    modules.test_for_missing_stem_and_pattern(data)
    modules.test_for_wrong_patterns(inflection_table_index, data)
    modules.test_for_differences_in_stem_and_pattern(data)
    modules.test_if_inflections_exist_suttas(data)
    modules.generate_changed_inflected_forms(data)
    diff = modules.combine_old_and_new_dataframes()
    modules.export_inflections_to_pickle(diff)
    modules.make_list_of_all_inflections()
    modules.make_list_of_all_inflections_no_meaning(data)
    # higlight
    modules.make_list_of_all_inflections_only_in_class(data)
    # red
    modules.make_list_of_all_inflections_already_in(data)
    # green
    modules.make_list_of_all_inflections_potential(dps_df=data, class_file_name=class_file_name)
    # blue
    sutta_file, commentary_file = modules.read_and_clean_sutta_text()
    modules.make_comparison_table(sutta_file, commentary_file)
    modules.html_find_and_replace(sutta_file)
    modules.write_html(sutta_file)
    modules.open_in_browser(sutta_file)


main()
