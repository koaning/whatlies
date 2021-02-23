from pathlib import Path

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from whatlies.language import DIETLanguage


@pytest.mark.rasa
def test_sklearn_pipeline_works():
    lang = DIETLanguage(
        model_path=next(Path("tests/rasa-test-demo/models").glob("*.tar.gz"))
    )
    pipe = Pipeline([("embed", lang), ("model", LogisticRegression())])

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "this is a bad post",
        "i dislike this article",
        "this is not well written",
    ]
    y = np.array([1, 1, 1, 0, 0, 0, 0])

    pipe.fit(X, y)
    assert pipe.predict(X).shape[0] == 7

    preprocess = FeatureUnion([("dense", lang), ("sparse", CountVectorizer())])

    assert preprocess.fit_transform(X).shape[0] == 7
