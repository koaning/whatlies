from typing import Union, List

import tensorflow_text
import tensorflow as tf
import tensorflow_hub as tfhub

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language.common import SklearnTransformerMixin, HiddenPrints


class ConveRTLanguage(SklearnTransformerMixin):
    """
    This object is used to fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a
    [ConveRT](https://github.com/PolyAI-LDN/polyai-models) model.
    This object is meant for retreival, not plotting.

    Important:
        This object will automatically download a large file if it is not cached yet.

        Also note that this language model does not contain a vocabulary, so it cannot be used
        to retreive similar tokens. Instead, you can create an embedding set by using this model
        and then perform similarity queries from there.

    Arguments:
        model_id: identifier used for loading the corresponding TFHub module, which could be one of `'convert`, `'convert-multi-context'` or `'convert-ubuntu'`.
            Each one of these correspond to a different model as described in [ConveRT manual](https://github.com/PolyAI-LDN/polyai-models#models).
        signature: the TFHub signature of the model, which could be one of `'default'`, `'encode_context'`, `'encode_response'` or `'encode_sequence'`.
            Note that `'encode_context'` is not currently supported with `'convert-multi-context'` or `'convert-ubuntu'` models.

    **Usage**:

    ```python
    > from whatlies.language import ConveRTLanguage
    > lang = ConveRTLanguage()
    > lang['bank']
    > lang = ConveRTLanguage(model_id='convert-multi-context', signature='encode_sequence')
    > lang[['bank of the river', 'money on the bank', 'bank']]
    ```
    """

    MODEL_URL = {
        "convert": "http://models.poly-ai.com/convert/v1/model.tar.gz",
        "convert-multi-context": "http://models.poly-ai.com/multi_context_convert/v1/model.tar.gz",
        "convert-ubuntu": "http://models.poly-ai.com/ubuntu_convert/v1/model.tar.gz",
    }

    MODEL_SIGNATURES = [
        "default",
        "encode_context",
        "encode_response",
        "encode_sequence",
    ]

    def __init__(self, model_id: str = "convert", signature: str = "default") -> None:
        if model_id not in self.MODEL_URL:
            raise ValueError(
                f"The `model_id` value should be one of {list(self.MODEL_URL.keys())}"
            )
        if signature not in self.MODEL_SIGNATURES:
            raise ValueError(
                f"The `signature` value should be one of {self.MODEL_SIGNATURES}"
            )
        if signature == "encode_context" and model_id in [
            "convert-multi-context",
            "convert-ubuntu",
        ]:
            raise NotImplementedError(
                "Currently 'encode_context' signature is not support with multi-context and ubuntu models."
            )
        self.model_id = model_id
        self.signature = signature

        with HiddenPrints():
            self.module = tfhub.load(self.MODEL_URL[self.model_id])
            self.model = self.module.signatures[self.signature]

    def __getitem__(
        self, query: Union[str, List[str]]
    ) -> Union[Embedding, EmbeddingSet]:
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**

        ```python
        > from whatlies.language import ConveRTLanguage
        > lang = ConveRTLanguage()
        > lang['bank']
        > lang = ConveRTLanguage()
        > lang[['bank of the river', 'money on the bank', 'bank']]
        ```
        """
        if isinstance(query, str):
            query_tensor = tf.convert_to_tensor([query])
            encoding = self.model(query_tensor)
            if self.signature == "encode_sequence":
                vec = encoding["sequence_encoding"].numpy().sum(axis=1)[0]
            else:
                vec = encoding["default"].numpy()[0]
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])
