import pytest

from whatlies.language import TFIDFVectorLanguage


@pytest.fixture
def lang():
    lang = TFIDFVectorLanguage(n_components=3, ngram_range=(1, 2), analyzer="char")
    return lang.fit_manual(
        ["pizza", "pizzas", "firehouse", "firehydrant", "cat", "dog"]
    )


def test_basic_docs_usage1(lang):
    embset = lang[["pizza", "pizzas", "firehouse", "firehydrant"]]
    assert embset.to_dataframe().shape == (4, 3)


def test_basic_docs_usage2(lang):
    embset = lang[["piza", "pizza", "pizzaz", "fyrehouse", "firehouse", "fyrehidrant"]]
    assert embset.to_dataframe().shape == (6, 3)


def test_basic_docs_usage3(lang):
    embedding = lang["piza"]
    assert embedding.name == "piza"
    assert embedding.vector.shape == (3,)


def test_score_similar(lang):
    assert len(lang.score_similar("cat", 6)) == 6


def test_value_errors1(lang):
    with pytest.raises(ValueError):
        _ = lang[""]


def test_value_errors2(lang):
    with pytest.raises(ValueError):
        lang.fit_manual(["", "cat", "dog"])


def test_retreival1(lang):
    assert lang.score_similar("doggg", n=1)[0][0].name == "dog"


def test_retreival2(lang):
    assert len(lang.score_similar("doggg", n=5)) == 5


def test_retreival_error(lang):
    with pytest.raises(ValueError):
        lang.score_similar("doggg", n=50)
