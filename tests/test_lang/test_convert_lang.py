import pytest

from whatlies.language import ConveRTLanguage


@pytest.fixture
def lang():
    lang = ConveRTLanguage()
    return lang


def test_basic_docs_usage1(lang):
    embset = lang[["bank", "money on the bank", "bank of the river"]]
    assert len(embset) == 3
    assert embset["bank"].vector.shape == (512,)
