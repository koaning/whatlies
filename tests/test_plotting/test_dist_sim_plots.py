from whatlies.language import SpacyLanguage

import pytest


@pytest.fixture()
def embset():
    lang = SpacyLanguage("en_core_web_sm")
    names = [
        "red",
        "blue",
        "green",
        "yellow",
        "cat",
        "dog",
        "mouse",
        "rat",
        "bike",
        "car",
    ]
    return lang[names]


def test_plot_distance_raises_error(embset):
    with pytest.raises(ValueError):
        embset.plot_distance(metric="dinosaurhead")


def test_plot_similarity_raises_error(embset):
    with pytest.raises(ValueError):
        embset.plot_distance(metric="dinosaurhead")
