import numpy as np

from whatlies.language import TFHubLanguage, UniversalSentenceLanguage


def test_same_results():
    use_lang = UniversalSentenceLanguage("multi", 3)
    tf_lang = TFHubLanguage(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    )
    assert np.allclose(use_lang["hello world"].vector, tf_lang["hello world"].vector)
