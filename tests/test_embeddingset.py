from operator import add, rshift, sub, or_

import pytest
import numpy as np
from spacy.vocab import Vocab
from spacy.language import Language

from whatlies import Embedding, EmbeddingSet
from whatlies.language import SpacyLanguage


@pytest.fixture()
def lang():
    vector_data = {
        k: np.random.normal(0, 1, (2,))
        for k in ["red", "blue", "cat", "dog", "green", "purple"]
    }
    vector_data["cat"] += 21.0
    vector_data["dog"] += 20.0
    vocab = Vocab(strings=vector_data.keys())
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


def test_embeddingset_creation():
    foo = Embedding("foo", [0, 1])
    bar = Embedding("bar", [1, 1])

    emb = EmbeddingSet(foo)
    assert len(emb) == 1
    assert "foo" in emb
    emb = EmbeddingSet(foo, bar)
    assert len(emb) == 2
    assert "foo" in emb
    assert "bar" in emb
    emb = EmbeddingSet({"foo": foo})
    assert len(emb) == 1
    assert "foo" in emb


@pytest.mark.parametrize("operator", [add, rshift, sub, or_])
def test_artificial_embset(lang, operator):
    emb = lang[["red", "blue", "orange"]]
    v1 = operator(emb["red"], emb["blue"])
    v2 = operator(lang["red"], lang["blue"])
    assert np.array_equal(v1.vector, v2.vector)


def test_merge_basic(lang):
    emb1 = lang[["red", "blue", "orange"]]
    emb2 = lang[["pink", "purple", "brown"]]
    assert len(emb1.merge(emb2)) == 6


def test_average(lang):
    emb = lang[["red", "blue", "orange"]]
    av = emb.average()
    assert av.name == "EmbSet.average()"
    v1 = av.vector
    v2 = (lang["red"].vector + lang["blue"].vector + lang["orange"].vector) / 3
    assert np.array_equal(v1, v2)


def test_to_x_y():
    foo = Embedding("foo", [0.1, 0.3])
    bar = Embedding("bar", [0.7, 0.2])
    buz = Embedding("buz", [0.1, 0.9])
    bla = Embedding("bla", [0.2, 0.8])

    emb1 = EmbeddingSet(foo, bar).add_property("label", lambda d: "group-one")
    emb2 = EmbeddingSet(buz, bla).add_property("label", lambda d: "group-two")
    emb = emb1.merge(emb2)

    X, y = emb.to_X_y(y_label="label")
    assert X.shape == emb.to_X().shape == (4, 2)
    assert list(y) == ["group-one", "group-one", "group-two", "group-two"]


def test_embset_similar_simple_len(lang):
    emb = lang[["red", "blue", "orange"]]
    assert len(emb.embset_similar("red", 1)) == 1
    assert len(emb.embset_similar("red", 2)) == 2


def test_embset_similar_simple_contains(lang):
    emb = lang[["red", "blue", "orange", "cat", "dog"]]
    subset_cat = emb.embset_similar("cat", 2, metric="euclidean").embeddings.keys()
    assert "cat" in subset_cat
    assert "dog" in subset_cat


def test_embset_similar_simple_distance(lang):
    emb = lang[["red", "blue", "orange", "cat", "dog"]]
    emb_red, score_red = emb.score_similar("red", 5)[0]
    assert np.isclose(score_red, 0.0, atol=0.0001)


def test_embset_raise_value_error_n(lang):
    emb = lang[["red", "blue", "orange", "cat", "dog"]]
    with pytest.raises(ValueError):
        emb.score_similar("red", 10)


def test_embset_raise_value_error_emb(lang):
    emb = lang[["red", "blue", "orange", "cat", "dog"]]
    with pytest.raises(ValueError):
        emb.score_similar("dinosaurhead", 1)


def test_corrplot_raise_error(lang):
    with pytest.raises(ValueError):
        emb = lang[["red", "blue", "orange", "pink", "purple", "brown"]]
        emb.plot_correlation(metric="dinosaurhead")


def test_filter(lang):
    emb = lang[["red", "blue", "orange", "pink", "purple", "brown"]]
    assert len(emb) == 6
    assert len(emb.filter(lambda e: "pink" not in e.name)) == 5
    assert len(emb.filter(lambda e: "pink" in e.name)) == 1


def test_pipe(lang):
    embset = lang[["red", "blue", "orange", "pink", "purple", "brown"]]
    assert embset.pipe(len) == 6
