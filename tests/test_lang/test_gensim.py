import pytest

from whatlies.language import GensimLanguage


@pytest.fixture()
def lang():
    return GensimLanguage("tests/cache/custom_gensim_vectors.kv")


def test_missing_retreival(lang):
    assert lang["doesnotexist"].vector.shape == (10,)


def test_spaces_retreival(lang):
    assert lang["graph trees"].vector.shape == (10,)
    assert lang["graph trees dog"].vector.shape == (10,)


def test_single_token_words(lang):
    assert lang["computer"].vector.shape == (10,)
    assert len(lang[["red", "blue"]]) == 2


def test_similar_retreival(lang):
    assert len(lang.score_similar("hi", 10)) == 10
    assert len(lang.embset_similar("hi", 10)) == 10
