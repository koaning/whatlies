import pytest

from whatlies.language import BytePairLanguage


@pytest.fixture()
def lang():
    return BytePairLanguage("en", vs=1000, dim=25, cache_dir="tests/cache")


def test_single_token_words(lang):
    assert lang["red"].vector.shape == (25,)
    assert len(lang[["red", "blue"]]) == 2


def test_similar_retreival(lang):
    assert len(lang.score_similar("hi", 10)) == 10
    assert len(lang.embset_similar("hi", 10)) == 10


@pytest.mark.parametrize("item", [2, 0.12341])
def test_raise_error(lang, item):
    with pytest.raises(ValueError):
        _ = lang[item]
