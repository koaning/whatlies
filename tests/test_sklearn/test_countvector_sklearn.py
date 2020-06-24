import pytest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from whatlies.language import CountVectorLanguage


@pytest.mark.parametrize("components", range(1, 6))
def test_sklearn_pipeline_works(components):
    lang = CountVectorLanguage(n_components=components)
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


def test_sklearn_feature_union_works():
    lang = CountVectorLanguage(n_components=2)
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
