import pytest

from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca


words = [
    "prince",
    "princess",
    "nurse",
    "doctor",
    "banker",
    "man",
    "woman",
    "cousin",
    "neice",
    "king",
    "queen",
    "dude",
    "guy",
    "gal",
    "fire",
    "dog",
    "cat",
    "mouse",
    "red",
    "bluee",
    "green",
    "yellow",
    "water",
    "person",
    "family",
    "brother",
    "sister",
]


@pytest.fixture
def embset():
    lang = SpacyLanguage("en_core_web_md")
    return lang[words]


def test_basic_dimensions_3d_chart(embset):
    ax = embset.transform(Pca(3)).plot_3d(annot=True)
    assert ax.xaxis.get_label_text() == "Dimension 0"
    assert ax.yaxis.get_label_text() == "Dimension 1"
    assert ax.zaxis.get_label_text() == "Dimension 2"
    assert [t.get_text() for t in ax.texts] == words


def test_named_dimensions_3d_chart(embset):
    ax = embset.transform(Pca(3)).plot_3d("king", "queen", "prince", annot=True)
    assert ax.xaxis.get_label_text() == "king"
    assert ax.yaxis.get_label_text() == "queen"
    assert ax.zaxis.get_label_text() == "prince"
    assert [t.get_text() for t in ax.texts] == words
