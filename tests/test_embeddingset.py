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


def test_embset_creation_error():
    foo = Embedding("foo", [0, 1])
    # This vector has a different dimension. No bueno.
    bar = Embedding("bar", [1, 1, 2])
    with pytest.raises(ValueError):
        EmbeddingSet(foo, bar)


def test_embset_creation_warning():
    foo = Embedding("foo", [0, 1])
    # This vector has the same name dimension. Dangerzone.
    bar = Embedding("foo", [1, 2])
    with pytest.raises(Warning):
        EmbeddingSet(foo, bar)


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


def test_filter(lang):
    emb = lang[["red", "blue", "orange", "pink", "purple", "brown"]]
    assert len(emb) == 6
    assert len(emb.filter(lambda e: "pink" not in e.name)) == 5
    assert len(emb.filter(lambda e: "pink" in e.name)) == 1


def test_pipe(lang):
    embset = lang[["red", "blue", "orange", "pink", "purple", "brown"]]
    assert embset.pipe(len) == 6


def test_to_names_X(lang):
    words = ["red", "blue", "dog"]
    embset = lang[words]
    names, X = embset.to_names_X()
    assert names == words
    assert np.array_equal(X, embset.to_X())


def test_from_names_X():
    names = ["foo", "bar", "buz"]
    X = [
        [1.0, 2],
        [3, 4.0],
        [0.5, 0.6],
    ]
    embset = EmbeddingSet.from_names_X(names, X)
    assert "foo" in embset
    assert len(embset) == 3
    assert np.array_equal(embset.to_X(), np.array(X))

    names = names[:2]
    with pytest.raises(ValueError, match="The number of given names"):
        EmbeddingSet.from_names_X(names, X)


def test_ndim(lang):
    embset = lang[["red", "blue", "dog"]]
    assert embset.ndim == 2


def test_compare_against(lang):
    embset = lang[["red", "blue", "cat"]]
    compared = embset.compare_against(lang["green"])
    true_values = np.array(
        [
            embset["red"] > lang["green"],
            embset["blue"] > lang["green"],
            embset["cat"] > lang["green"],
        ]
    )
    assert np.array_equal(compared, true_values)

    # Test with custom mapping function
    compared = embset.compare_against("cat", mapping=np.dot)
    true_values = np.array(
        [
            np.dot(embset["red"].vector, lang["cat"].vector),
            np.dot(embset["blue"].vector, lang["cat"].vector),
            np.dot(embset["cat"].vector, lang["cat"].vector),
        ]
    )
    assert np.array_equal(compared, true_values)

    # Test with non-existent name or invalid mapping
    with pytest.raises(KeyError):
        embset.compare_against("purple")
    with pytest.raises(ValueError, match="Unrecognized mapping value/type."):
        embset.compare_against(lang["green"], mapping="cosine")


def test_add_property():
    foo = Embedding("foo", [0.1, 0.3, 0.10])
    bar = Embedding("bar", [0.7, 0.2, 0.11])
    emb = EmbeddingSet(foo, bar)
    emb_with_property = emb.add_property("prop_a", lambda d: "prop-one")
    assert all([e.prop_a == "prop-one" for e in emb_with_property])


def test_assign_base():
    foo = Embedding("foo", [0.1, 0.3, 0.10])
    bar = Embedding("bar", [0.7, 0.2, 0.11])
    emb = EmbeddingSet(foo, bar)
    emb_with_property = emb.assign(
        prop_a=lambda d: "prop-one", prop_b=lambda d: "prop-two"
    )
    assert all([e.prop_a == "prop-one" for e in emb_with_property])
    assert all([e.prop_b == "prop-two" for e in emb_with_property])


def test_assign_literal_values():
    foo = Embedding("foo", [0.1, 0.3, 0.10])
    bar = Embedding("bar", [0.7, 0.2, 0.11])
    emb = EmbeddingSet(foo, bar)
    emb_with_property = emb.assign(prop_a="prop-one", prop_b=1)
    assert all([e.prop_a == "prop-one" for e in emb_with_property])
    assert all([e.prop_b == 1 for e in emb_with_property])


def test_assign_arrays():
    foo = Embedding("foo", [0.1, 0.3, 0.10])
    bar = Embedding("bar", [0.7, 0.2, 0.11])
    emb = EmbeddingSet(foo, bar)
    emb_with_property = emb.assign(prop_a=["a", "b"], prop_b=np.array([1, 2]))
    assert emb_with_property["foo"].prop_a == "a"
    assert emb_with_property["bar"].prop_a == "b"
    assert emb_with_property["foo"].prop_b == 1
    assert emb_with_property["bar"].prop_b == 2


def test_assign_arrays_raise_error():
    foo = Embedding("foo", [0.1, 0.3, 0.10])
    bar = Embedding("bar", [0.7, 0.2, 0.11])
    emb = EmbeddingSet(foo, bar)
    with pytest.raises(ValueError):
        emb.assign(prop_a=["a", "b"], prop_b=np.array([1, 2, 3]))
