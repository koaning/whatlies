import pytest
import numpy as np
from sklearn.utils import estimator_checks

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import SpacyLanguage


transformer_checks = (
    estimator_checks.check_transformer_data_not_an_array,
    estimator_checks.check_transformer_general,
    estimator_checks.check_transformers_unfitted,
)


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


@pytest.mark.parametrize('text', [("red red", "blue red"), ("red", "green", "blue"), ("dog", "cat")])
def test_check_sizes(color_lang, text):
    X = [text]
    assert color_lang.fit(X).transform(X).shape == (len(text), 2)
    assert color_lang.fit_transform(X).shape == (len(text), 2)


@pytest.mark.parametrize("test_fn", transformer_checks)
def test_estimator_checks(test_fn, color_lang):
    test_fn("spacy_lang", color_lang)
