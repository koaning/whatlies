from typing import Union

from ._tfhub_lang import TFHubLanguage


def UniversalSentenceLanguage(variant: str = "base", version: Union[int, None] = None):
    """
    Retreive a [universal sentence encoder](https://tfhub.dev/google/collections/universal-sentence-encoder/1) model from tfhub.

    You can download specific versions for specific variants. The variants that we support are listed below.

    - `"base"`: the base variant (915MB) [link](https://tfhub.dev/google/universal-sentence-encoder/4)
    - `"large"`: the large variant (523MB) [link](https://tfhub.dev/google/universal-sentence-encoder-large/5)
    - `"qa"`: the variant based on question/answer (528MB) [link](https://tfhub.dev/google/universal-sentence-encoder-qa/3)
    - `"multi"`: the multi-language variant (245MB) [link](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)
    - `"multi-large"`: the large multi-language variant (303MB) [link](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)
    - `"multi-qa"`: the multi-language qa variant (310MB) [link](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3)

    TFHub reports that the multi-language models support Arabic, Chinese-simplified, Chinese-traditional,
    English, French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Spanish, Thai, Turkish and Russian.

    Important:
        This object will automatically download a large file if it is not cached yet.

        This language model does not contain a vocabulary, so it cannot be used
        to retreive similar tokens. Use an `EmbeddingSet` instead.

        This language backend might require you to manually install extra dependencies
        unless you installed via either;

        ```
        pip install whatlies[tfhub]
        pip install whatlies[all]
        ```

    Arguments:
        variant: select a specific variant
        version: select a specific version, if kept `None` we'll assume the most recent version
    """
    urls = {
        "base": "https://tfhub.dev/google/universal-sentence-encoder/",
        "large": "https://tfhub.dev/google/universal-sentence-encoder-large/",
        "qa": "https://tfhub.dev/google/universal-sentence-encoder-qa/",
        "multi": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/",
        "multi-large": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/",
        "multi-qa": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
    }

    versions = {
        "base": 4,
        "large": 5,
        "qa": 3,
        "multi": 3,
        "multi-large": 3,
        "multi-qa": 3,
    }

    version = versions[variant] if not version else version
    url = urls[variant] + str(version)
    return TFHubLanguage(url=url)
