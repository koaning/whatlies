import pytest

from whatlies.language import TFHubLanguage


@pytest.fixture
def lang(request):
    return TFHubLanguage(**request.param)


@pytest.mark.parametrize(
    "lang, expected_shape",
    [
        (
            {"url": "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"},
            (20,),
        ),
        (
            {
                "url": "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
            },
            (20,),
        ),
    ],
    indirect=["lang"],
)
def test_basic_usage(lang, expected_shape):
    emb = lang["a test sentence"]
    assert emb.vector.shape == expected_shape
    emb = lang[["test", "a simple test sentence", "and another nice sentence"]]
    assert len(emb) == 3
    assert emb["test"].vector.shape == expected_shape
