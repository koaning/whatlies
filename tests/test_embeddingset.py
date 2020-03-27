from operator import add, rshift, sub, or_

import pytest
import numpy as np

from whatlies import Embedding, EmbeddingSet
from whatlies.language import SpacyLanguage

lang = SpacyLanguage("en_core_web_sm")


@pytest.mark.parametrize("operator", [add, rshift, sub, or_])
def test_artificial_embset(operator):
    emb = lang[["red", "blue", "orange"]]
    v1 = operator(emb["red"], emb["blue"])
    v2 = operator(lang["red"], lang["blue"])
    assert np.array_equal(v1.vector, v2.vector)


# def test_operator_name():
#     emb = lang[['red', 'blue', 'orange']]
# assert str(emb + emb["red"]) == "(Emb + Emb[red])"
# assert str(emb - emb["red"]) == "(Emb - Emb[red])"
# assert str(emb | (emb["red"] - emb["blue"])) == "(Emb | (Emb[red] - Emb[blue]))"
# assert str((emb | emb["red"]) - emb["blue"]) == "((Emb | Emb[red]) - Emb[blue])"


def test_merge_basic():
    emb1 = lang[["red", "blue", "orange"]]
    emb2 = lang[["pink", "purple", "brown"]]
    assert len(emb1.merge(emb2)) == 6


def test_average():
    emb = lang[["red", "blue", "orange"]]
    av = emb.average()
    assert av.name == "Emb.average()"
    v1 = av.vector
    v2 = (lang["red"].vector + lang["blue"].vector + lang["orange"].vector) / 3
    assert np.array_equal(v1, v2)


def test_to_x_y():
    foo = Embedding("foo", [0.1, 0.3])
    bar = Embedding("bar", [0.7, 0.2])
    buz = Embedding("buz", [0.1, 0.9])
    bla = Embedding("bla", [0.2, 0.8])

    emb1 = EmbeddingSet(foo, bar).add_property("label", lambda d: 'group-one')
    emb2 = EmbeddingSet(buz, bla).add_property("label", lambda d: 'group-two')
    emb = emb1.merge(emb2)

    X, y = emb.to_X_y(y_label='label')
    assert X.shape == emb.to_X().shape == (4, 2)
    assert list(y) == ['group-one', 'group-one', 'group-two', 'group-two']
