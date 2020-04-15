import pytest

from whatlies.language import SpacyLanguage
from whatlies.transformers import Umap, Pca, Noise, AddRandom

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

lang = SpacyLanguage("en_core_web_sm")
emb = lang[words]


@pytest.mark.parametrize(
    "transformer,extra_size",
    zip(
        [
            Umap(2),
            Umap(3),
            Pca(2),
            Pca(3),
            Noise(0.1),
            Noise(0.01),
            AddRandom(n=4),
            AddRandom(n=1),
            lambda d: d | (d["man"] - d["woman"]),
        ],
        [2, 3, 2, 3, 0, 0, 4, 1, 0],
    ),
)
def test_transformations_new_size(transformer, extra_size):
    emb_new = emb.transform(transformer)
    assert len(emb_new) == len(emb) + extra_size


@pytest.mark.parametrize(
    "transformer",
    [
        Umap(2),
        Pca(2),
        Noise(0.1),
        AddRandom(n=4),
        lambda d: d | (d["man"] - d["woman"]),
    ],
)
def test_transformations_keep_props(transformer):
    emb_new = emb.add_property("group", lambda d: "one").transform(transformer)
    for w in words:
        assert hasattr(emb_new[w], "group")
