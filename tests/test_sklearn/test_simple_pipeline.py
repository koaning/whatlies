import pytest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from whatlies.language import (
    FasttextLanguage,
    SpacyLanguage,
    GensimLanguage,
    BytePairLanguage,
    TFHubLanguage,
    HFTransformersLanguage,
    SentenceTFMLanguage,
)


backends = [
    SpacyLanguage("tests/custom_test_lang/"),
    FasttextLanguage("tests/custom_fasttext_model.bin"),
    BytePairLanguage("en", vs=1000, dim=25, cache_dir="tests/cache"),
    GensimLanguage("tests/cache/custom_gensim_vectors.kv"),
    HFTransformersLanguage("sshleifer/tiny-gpt2", framework="tf"),
    TFHubLanguage("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"),
    SentenceTFMLanguage("distilbert-base-nli-stsb-mean-tokens"),
]


@pytest.mark.parametrize("lang", backends)
def test_sklearn_pipeline_works(lang):
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
