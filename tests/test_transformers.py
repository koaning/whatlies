import pytest
from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import SpacyLanguage
from whatlies.transformers import Umap, Pca, Noise, AddRandom


vocab = Vocab().from_disk("tests/custom_test_vocab/")
words = list(vocab.strings)
lang = SpacyLanguage(nlp=Language(vocab=vocab, meta={"lang": "en"}))
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
