import pytest

import numpy as np

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
    "blue",
    "green",
    "yellow",
    "water",
    "person",
    "family",
    "brother",
    "sister",
]

# I'm loading in the spaCy model globally because it is much faster this way.
lang = SpacyLanguage("en_core_web_sm")


@pytest.fixture
def embset():
    return lang[words]


def test_set_title_works(embset):
    ax = embset.plot_3d(annot=True, title="foobar")
    assert ax.title._text == "foobar"


def test_correct_points_plotted(embset):
    embset_plt = embset.transform(Pca(3))
    ax = embset_plt.plot_3d(annot=True)
    offset = ax.collections[0]._offsets3d
    assert np.all(np.array(offset).T == embset_plt.to_X())


def test_correct_points_plotted_mapped(embset):
    embset_plt = embset.transform(Pca(3))
    ax = embset_plt.plot_3d("king", "red", "dog", annot=True)
    offset = ax.collections[0]._offsets3d
    king, red, dog = [v for v in np.array(offset)]

    assert np.all(king == np.array([embset_plt[w] > embset_plt["king"] for w in words]))
    assert np.all(red == np.array([embset_plt[w] > embset_plt["red"] for w in words]))
    assert np.all(dog == np.array([embset_plt[w] > embset_plt["dog"] for w in words]))


def test_basic_dimensions_3d_chart(embset):
    embset_plt = embset.transform(Pca(3))
    ax = embset_plt.plot_3d(annot=True, title="foobar")
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


def test_named_dimensions_3d_chart_rename(embset):
    ax = embset.transform(Pca(3)).plot_3d(
        "king", "queen", "prince", annot=True, x_label="x", y_label="y"
    )
    assert ax.xaxis.get_label_text() == "x"
    assert ax.yaxis.get_label_text() == "y"
    assert ax.zaxis.get_label_text() == "prince"
    assert [t.get_text() for t in ax.texts] == words
