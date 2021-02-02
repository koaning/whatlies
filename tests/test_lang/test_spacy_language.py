import pytest
import numpy as np

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import SpacyLanguage


@pytest.fixture()
def color_lang():
    vector_data = {
        "red": np.array([1.0, 0.0]),
        "green": np.array([0.5, 0.5]),
        "blue": np.array([0.0, 1.0]),
        "purple": np.array([0.0, 1.0]),
    }

    vocab = Vocab(strings=list(vector_data.keys()))
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


def test_basic_usage(color_lang):
    queries = [
        "green is blue and yellow",
        "purple is red and blue",
        "purple isn't same as red!",
        "red and blue is a like a glue!",
    ]
    emb = color_lang[queries]
    assert len(emb) == 4
    assert emb[queries[0]].name == "green is blue and yellow"
    assert emb[queries[0]].vector.shape == (2,)


def test_score_similar_one(color_lang):
    scores = color_lang.score_similar("blue", n=2, prob_limit=None, lower=False)
    print(scores)
    assert all([s[1] == 0 for s in scores])
    assert "blue" in [s[0].name for s in scores]
    assert "purple" in [s[0].name for s in scores]


def test_single_token_words(color_lang):
    # test for issue here: https://github.com/RasaHQ/whatlies/issues/5
    assert len(color_lang["red"].vector) > 0
