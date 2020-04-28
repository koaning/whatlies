import pytest
import numpy as np

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language.language import SpacyLanguage, _selected_idx_spacy


@pytest.fixture()
def color_lang():
    vector_data = {"red": np.array([1.0, 0.0]),
                   "green": np.array([0.5, 0.5]),
                   "blue": np.array([0.0, 1.0]),
                   "purple": np.array([0.0, 1.0])}

    vocab = Vocab(strings=vector_data.keys())
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


def test_score_similar_one(color_lang):
    scores = color_lang.score_similar("blue", n=2, prob_limit=None, lower=False)
    print(scores)
    assert all([s[1] == 0 for s in scores])
    assert "blue" in [s[0].name for s in scores]
    assert "purple" in [s[0].name for s in scores]


@pytest.mark.parametrize(
    "string, array",
    zip(
        ["red", "red green [blue] purple", "green [red blue] pink"],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])],
    ),
)
def test_lang_retreival(color_lang, string, array):
    assert np.isclose(color_lang[string].vector, array).all()


def test_single_token_words(color_lang):
    # test for issue here: https://github.com/RasaHQ/whatlies/issues/5
    assert np.sum(color_lang["red"].vector) > 0


@pytest.mark.parametrize(
    "triplets",
    zip(
        ["red", "red green", "red green [blue] purple", "red [green blue] pink"],
        [0, 0, 2, 1],
        [1, 2, 3, 3],
    ),
)
def test_select_idx_func(triplets):
    string, start, end = triplets
    assert _selected_idx_spacy(string) == (start, end)
