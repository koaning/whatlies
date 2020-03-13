import pytest
import numpy as np
from whatlies.language import SpacyLanguage


def test_always_true():
    assert True


@pytest.mark.parametrize("lang_name", ['en_core_web_sm', 'en_core_web_md', 'en_trf_robertabase_lg'])
def test_single_token_words(lang_name):
    # test for https://github.com/RasaHQ/whatlies/issues/5
    lang = SpacyLanguage(lang_name)
    assert np.sum(lang['red'].vector) > 0


@pytest.mark.parametrize("lang_name", ['en_trf_robertabase_lg'])
def test_bert_selection(lang_name):
    # test for https://github.com/RasaHQ/whatlies/issues/6
    lang = SpacyLanguage(lang_name)
    same = np.array_equal(
        lang.nlp("Going to the [store]"),
        lang.nlp("[Store] this in the drawer")
    )
    assert not same
