import pytest

from whatlies.language import BPEmbLang


@pytest.fixture()
def lang():
    return BPEmbLang("en")


def test_single_token_words(lang):
    assert lang["red"].vector.shape == (100, )
    assert len(lang[["red", "blue"]]) == 2
