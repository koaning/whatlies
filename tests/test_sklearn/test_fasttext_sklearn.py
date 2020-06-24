import pytest

from whatlies.language import FasttextLanguage

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture()
def lang():
    return FasttextLanguage("tests/custom_fasttext_model.bin")


@pytest.mark.parametrize(
    "text", [("red red", "blue red"), ("red", "green", "blue"), ("dog", "cat")]
)
def test_check_sizes(lang, text):
    X = text
    assert lang.fit(X).transform(X).shape == (len(text), 10)
    assert lang.fit_transform(X).shape == (len(text), 10)


def test_get_params(lang):
    assert "model" in lang.get_params().keys()
    assert "size" in lang.get_params().keys()


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
    test_fn("spacy_lang", FasttextLanguage("tests/custom_fasttext_model.bin"))


def test_sklearn_pipeline_works(lang):
    pipe = Pipeline([("embed", lang), ("model", LogisticRegression())])

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written",
    ]
    y = np.array([1, 1, 1, 0, 0, 0])

    pipe.fit(X, y)
    assert pipe.predict(X).shape[0] == 6


def test_sklearn_feature_union_works(lang):
    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written",
    ]

    preprocess = FeatureUnion([("dense", lang), ("sparse", CountVectorizer())])

    assert preprocess.fit_transform(X).shape[0] == 6
