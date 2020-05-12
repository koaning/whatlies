import pytest
import numpy as np
from sklearn.utils import estimator_checks

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import FasttextLanguage


@pytest.fixture()
def lang():
    return FasttextLanguage('tests/custom_fasttext_model.bin')


@pytest.mark.parametrize('text', [("red red", "blue red"), ("red", "green", "blue"), ("dog", "cat")])
def test_check_sizes(lang, text):
    X = [text]
    assert lang.fit(X).transform(X).shape == (len(text), 10)
    assert lang.fit_transform(X).shape == (len(text), 10)


def test_get_params(lang):
    assert 'model' in lang.get_params().keys()
    assert 'size' in lang.get_params().keys()


checks = (
    # all of these checks fail because fasttext is not pickle-able (it's native c++ code)
    # estimator_checks.check_fit2d_predict1d,
    # estimator_checks.check_fit2d_1sample,
    # estimator_checks.check_fit2d_1feature,
    # estimator_checks.check_fit1d,
    # estimator_checks.check_get_params_invariance,
    # estimator_checks.check_set_params,
    # estimator_checks.check_dont_overwrite_parameters,
    # estimator_checks.check_transformers_unfitted,
    # estimator_checks.check_transformer_data_not_an_array,
    # estimator_checks.check_transformer_general,
    # estimator_checks.check_methods_subset_invariance,
    # estimator_checks.check_dict_unchanged,
)


@pytest.mark.parametrize("test_fn", checks)
def test_estimator_checks(test_fn):
    test_fn("spacy_lang", FasttextLanguage('tests/custom_fasttext_model.bin'))
