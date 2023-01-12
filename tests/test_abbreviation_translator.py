import pytest

from inflection_generator.abbreviation_translator import AbbreviationTranslator


@pytest.fixture
def abbrev():
    obj = AbbreviationTranslator('cyrl')
    obj.set_dict({'key': 'value'})
    return obj


def test_transalte_simple(abbrev):
    assert abbrev.translate_string('key') == 'value'
    assert abbrev.translate_string('key ') == 'value '
    assert abbrev.translate_string(' key ') == ' value '
    assert abbrev.translate_string(' key') == ' value'


def test_transalte_double(abbrev):
    assert abbrev.translate_string('keykey') == 'keykey'
    assert abbrev.translate_string('key key') == 'value value'
    assert abbrev.translate_string('key str key') == 'value str value'
    assert abbrev.translate_string('key ke') == 'value ke'

def test_transalte_no_entry(abbrev):
    assert abbrev.translate_string('eyke') == 'eyke'
    assert abbrev.translate_string('value') == 'value'
    assert abbrev.translate_string('ke ke') == 'ke ke'
