import numpy as np

from whatlies.language import TFHubLanguage, UniversalSentenceLanguage


def test_same_results():
    use_lang = UniversalSentenceLanguage("lite", 1)
    tf_lang = TFHubLanguage(
        "https://tfhub.dev/google/universal-sentence-encoder-lite/1"
    )
    assert np.allclose(use_lang["hello world"], tf_lang["hello world"])
