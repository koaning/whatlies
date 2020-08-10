import pytest
import numpy as np
from sklearn.utils import estimator_checks

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

    vocab = Vocab(strings=vector_data.keys())
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


@pytest.mark.parametrize(
    "text", [("red red", "blue red"), ("red", "green", "blue"), ("dog", "cat")]
)
def test_check_sizes(color_lang, text):
    assert color_lang.fit(text).transform(text).shape == (len(text), 2)
    assert color_lang.fit_transform(text).shape == (len(text), 2)


def test_get_params():
    assert "nlp" in SpacyLanguage("tests/custom_test_lang/").get_params().keys()


checks = (
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_transformers_unfitted,
    # these checks won't work because they assume text data
    # estimator_checks.check_fit2d_predict1d,
    # estimator_checks.check_dont_overwrite_parameters,
    # estimator_checks.check_fit2d_1sample,
    # estimator_checks.check_fit2d_1feature,
    # estimator_checks.check_transformer_data_not_an_array,
    # estimator_checks.check_transformer_general,
    # estimator_checks.check_methods_subset_invariance,
    # estimator_checks.check_dict_unchanged,
)


@pytest.mark.parametrize("test_fn", checks)
def test_estimator_checks(test_fn):
    test_fn("spacy_lang", SpacyLanguage("tests/custom_test_lang/"))
