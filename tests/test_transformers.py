import pytest

from whatlies.language import SpacyLanguage
from whatlies.transformers import Umap, Pca, Noise, AddRandom

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

lang = SpacyLanguage("en_core_web_sm")
emb = lang[words]


@pytest.mark.parametrize(
    "tfm",
    [Umap(2), Umap(3), Pca(2), Pca(3), Noise(0.1), Noise(0.01), AddRandom()],
)
def test_transformations_no_error(tfm):
    emb_new = emb.transform(tfm)
    assert len(emb_new) >= len(emb)
