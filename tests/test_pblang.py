import pytest

from whatlies.language import BPEmbLang


@pytest.fixture()
def lang():
    return BPEmbLang("en")


def test_single_token_words(lang):
    assert lang["red"].vector.shape == (100, )
    assert len(lang[["red", "blue"]]) == 2


@pytest.mark.parametrize("item", [2, .12341])
def test_raise_error(lang, item):
    with pytest.raises(ValueError):
        _ = lang[item]
