import argparse

from rich import print  # pylint: disable=redefined-builtin

from inflection_generator import modules
from inflection_generator.helpers import Kind, timeis


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=[i.name for i in Kind])
    parser.add_argument("--class-file-name", type=str, default='1')
    return parser


def generate(args: argparse.Namespace) -> None:
    print(f"{timeis()} ----------------------------------------")

    inflection_table_index = modules.create_inflection_table_index()
    inflection_table = modules.create_inflection_table_df()

    modules.test_inflection_pattern_changed(inflection_table_index, inflection_table)

    kind = Kind[args.kind]

    if kind is Kind.DPS:
        data = modules.create_dps_df()
    elif kind is Kind.SBS:
        data = modules.create_sbs_df(args.class_file_name)

    modules.test_for_missing_stem_and_pattern(data)
    modules.test_for_wrong_patterns(inflection_table_index, data)
    modules.test_for_differences_in_stem_and_pattern(data)
    modules.test_if_inflections_exist_dps(data)
    modules.test_if_inflections_exist_suttas(data)  # nu
    modules.generate_changed_inflected_forms(data)
    modules.combine_old_and_new_dataframes()

    table_generator = modules.InflectionTableGenerator(data, inflection_table_index, kind)
    table_generator.generate_html()

    modules.generate_inflections_in_table_list(data)
    modules.transcribe_new_inflections()
    modules.combine_old_and_new_translit_dataframes()
    modules.export_translit_to_pickle()
    modules.export_inflections_to_pickle()
    modules.delete_unused_inflection_patterns(inflection_table_index)
    modules.delete_old_pickle_files()
    modules.delete_unused_html_tables()
    modules.delete_unused_inflections()
    modules.delete_unused_inflections_translit()

    print(f"{timeis()} ----------------------------------------")


def suttas() -> None:
    modules.read_and_clean_sutta_text()
    #make_comparison_table()


def main() -> None:
    ARGS = get_argparser().parse_args()
    generate(ARGS)
