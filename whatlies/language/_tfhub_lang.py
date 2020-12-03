from typing import Union, List, Optional

import tensorflow_text  # noqa: F401
import tensorflow_hub as tfhub

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin


class TFHubLanguage(SklearnTransformerMixin):
    """
    This class provides the abitilty to load and use text-embedding models of Tensorflow Hub to
    retrieve [Embedding][whatlies.embedding.Embedding]s or [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from them.
    A list of supported models is available [here](https://tfhub.dev/s?module-type=text-embedding&tf-version=tf2);
    however, note that only those models which operate directly on raw text (i.e. don't require any pre-processing
    such as tokenization) are supported for the moment (e.g. models such as BERT or ALBERT are not supported).
    Further, the TF-Hub compatible models from other repositories (i.e. other than [tfhub.dev](https://tfhub.dev/))
    are also supported.

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

        Further, consider that this language model mainly supports TensorFlow 2.x models (i.e. TF2 `SavedModel` format);
        although, TensorFlow 1.x models might be supported to some extent as well (see
        [`hub.load`](https://www.tensorflow.org/hub/api_docs/python/hub/load) documentation as well as
        [model compatibility guide](https://www.tensorflow.org/hub/model_compatibility)).

    Arguments:
        url: The url or local directory path of the model.
        tags: A set of strings specifying the graph variant to use, if loading from a TF1 module.
            It is passed to `hub.load` function.
        signature: An optional signature of the model to use.

    **Usage**:

    ```python
    > from whatlies.language import TFHubLanguage
    > lang = TFHubLanguage("https://tfhub.dev/google/nnlm-en-dim50/2")
    > lang['today is a gift']
    > lang = TFHubLanguage("https://tfhub.dev/google/nnlm-en-dim50/2")
    > lang[['withdraw some money', 'take out cash', 'cash out funds']]
    ```
    """

    def __init__(
        self,
        url: str,
        tags: Optional[List[str]] = None,
        signature: Optional[str] = None,
    ) -> None:
        model = tfhub.load(url, tags=tags)
        if signature:
            model = model.signatures[signature]
        self.signature = signature
        self.url = url
        self.tags = tags
        self.model = model

    def __getitem__(
        self, query: Union[str, List[str]]
    ) -> Union[Embedding, EmbeddingSet]:
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**

        ```python
        > from whatlies.language import TFHubLanguage
        > lang = TFHubLanguage("https://tfhub.dev/google/nnlm-en-dim50/2")
        > lang['today is a gift']
        > lang = TFHubLanguage("https://tfhub.dev/google/nnlm-en-dim50/2")
        > lang[['withdraw some money', 'take out cash', 'cash out funds']]
        ```
        """
        if isinstance(query, str):
            return self._get_embedding(query)
        return EmbeddingSet(*[self._get_embedding(q) for q in query])

    def _get_embedding(self, query: str) -> Embedding:
        vec = self.model([query]).numpy()[0]
        return Embedding(query, vec)
