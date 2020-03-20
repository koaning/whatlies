from operator import add, rshift, sub, or_

import pytest
import numpy as np

from whatlies.language import SpacyLanguage
from whatlies.embeddingset import EmbeddingSet
from whatlies.embedding import Embedding

lang = SpacyLanguage("en_core_web_sm")


@pytest.mark.parametrize("operator", [add, rshift, sub, or_])
def test_artificial_embset(operator):
    emb = lang[['red', 'blue', 'orange']]
    v1 = operator(emb['red'], emb['blue'])
    v2 = operator(lang['red'], lang['blue'])
    assert np.array_equal(v1.vector, v2.vector)


def test_operator_name():
    emb = lang[['red', 'blue', 'orange']]
    assert str(emb + emb["red"]) == "(Emb + Emb[red])"
    assert str(emb - emb["red"]) == "(Emb - Emb[red])"
    assert str(emb | (emb["red"] - emb["blue"])) == "(Emb | (Emb[red] - Emb[blue]))"
    assert str((emb | emb["red"]) - emb["blue"]) == "((Emb | Emb[red]) - Emb[blue])"


def test_merge_basic():
    emb1 = lang[['red', 'blue', 'orange']]
    emb2 = lang[['pink', 'purple', 'brown']]
    assert len(emb1.merge(emb2)) == 6
