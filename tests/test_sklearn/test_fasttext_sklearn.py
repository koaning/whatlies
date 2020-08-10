import pytest

from whatlies.language import FasttextLanguage


@pytest.fixture()
def lang():
    return FasttextLanguage("tests/custom_fasttext_model.bin")


@pytest.mark.parametrize(
    "text", [("red red", "blue red"), ("red", "green", "blue"), ("dog", "cat")]
)
def test_check_sizes(lang, text):
    X = text
    assert lang.fit(X).transform(X).shape == (len(text), 10)
    assert lang.fit_transform(X).shape == (len(text), 10)


def test_get_params(lang):
    assert "model" in lang.get_params().keys()
    assert "size" in lang.get_params().keys()
