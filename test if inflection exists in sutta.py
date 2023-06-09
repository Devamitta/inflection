# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

import warnings

from inflection_generator import modules

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    # modules.convert_dpd_ods_to_csv()
    inflection_table_index = modules.create_inflection_table_index()
    inflection_table = modules.create_inflection_table_df()
    modules.test_inflection_pattern_changed(inflection_table_index, inflection_table)
    data = modules.create_dps_df()
    modules.test_for_missing_stem_and_pattern(data)
    modules.test_for_wrong_patterns(inflection_table_index, data)
    modules.test_for_differences_in_stem_and_pattern(data)
    modules.test_if_inflections_exist_suttas(data)
    modules.generate_changed_inflected_forms(data)
    diff = modules.combine_old_and_new_dataframes()
    modules.export_inflections_to_pickle(diff)
    modules.make_list_of_all_inflections()
    modules.make_list_of_all_inflections_no_meaning(data)
    modules.make_list_of_all_inflections_no_eg1(data)
    modules.make_list_of_all_inflections_no_eg2(data)
    modules.make_list_of_all_inflections_no_eg3(data)
    sutta_file, commentary_file = modules.read_and_clean_sutta_text()
    modules.make_comparison_table(sutta_file, commentary_file)
    modules.html_find_and_replace(sutta_file)
    modules.write_html(sutta_file)
    moudles.open_in_browser(sutta_file)


main()
