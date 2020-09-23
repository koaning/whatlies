import pytest

import numpy as np
from sklearn.preprocessing import normalize
from spacy.vocab import Vocab
from spacy.language import Language

from whatlies.language import SpacyLanguage
from whatlies.transformers import (
    Transformer,
    Umap,
    Pca,
    Noise,
    AddRandom,
    Tsne,
    OpenTsne,
    # Ivis,
    Normalizer,
)


vocab = Vocab().from_disk("tests/custom_test_vocab/")
words = list(vocab.strings)
lang = SpacyLanguage(nlp=Language(vocab=vocab, meta={"lang": "en"}))
emb = lang[words]

transformers = [
    Umap(2),
    Umap(3),
    Pca(2),
    Pca(3),
    Noise(0.1),
    Noise(0.01),
    AddRandom(n=4),
    AddRandom(n=1),
    lambda d: d | (d["man"] - d["woman"]),
    Tsne(2, n_iter=250),
    Tsne(3, n_iter=250),
    OpenTsne(2, n_iter=2),
    # Ivis(2, k=10, batch_size=10, epochs=10),
    # Ivis(3, k=10, batch_size=10, epochs=10),
    Normalizer(),
    Normalizer(feature=True),
]
extra_sizes = [0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0]
tfm_ids = [_.__class__.__name__ for _ in transformers]


@pytest.mark.parametrize(
    "transformer,extra_size", zip(transformers, extra_sizes), ids=tfm_ids
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
        Tsne(2, n_iter=250),
        OpenTsne(2, n_iter=1),
        # Ivis(2, k=10, batch_size=10, epochs=10),
        AddRandom(n=4),
        lambda d: d | (d["man"] - d["woman"]),
        Normalizer(),
    ],
)
def test_transformations_keep_props(transformer):
    emb_new = emb.add_property("group", lambda d: "one").transform(transformer)
    for w in words:
        assert hasattr(emb_new[w], "group")


@pytest.mark.parametrize(
    "transformer",
    [
        Umap(2),
        Pca(2),
        Noise(0.1),
        AddRandom(n=4),
        Tsne(2, n_iter=250),
        OpenTsne(2, n_iter=2),
        # Ivis(2, k=10, batch_size=10, epochs=10),
        Normalizer(),
    ],
)
def test_transformers_are_subclassed_properly(transformer):
    assert isinstance(transformer, Transformer)


def test_transformer_base_class():
    with pytest.raises(TypeError, match="Can't instantiate abstract class Transformer"):
        Transformer()


def test_normalizer_transformer():
    X = emb.to_X()
    normalized_emb = emb.transform(Normalizer())
    assert set(emb.embeddings.keys()) == set(normalized_emb.embeddings.keys())
    assert np.array_equal(normalized_emb.to_X(), normalize(X, norm="l1"))

    normalized_emb = emb.transform(Normalizer(norm="l2"))
    assert set(emb.embeddings.keys()) == set(normalized_emb.embeddings.keys())
    assert np.array_equal(normalized_emb.to_X(), normalize(X))

    normalized_emb = emb.transform(Normalizer(feature=True))
    assert set(emb.embeddings.keys()) == set(normalized_emb.embeddings.keys())
    assert np.array_equal(normalized_emb.to_X(), normalize(X, norm="l1", axis=0))
