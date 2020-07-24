from typing import Union, List
import tensorflow_text
import tensorflow as tf
import tensorflow_hub as tfhub


from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language.common import SklearnTransformerMixin, HiddenPrints


class ConveRTLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a keyed vector file.
    These files are generated from [ConveRT](https://github.com/PolyAI-LDN/polyai-models).
    This object is meant for retreival, not plotting.

    Important:
        This object will automatically download a large file if it is not cached yet.

        Also note that this language model does not contain a vocabulary so it cannot be used
        to retreive similar tokens. Instead you can create an embedding set and make do the
        similarity query from there.

    **Usage**:

    ```python
    > from whatlies.language import ConveRTLanguage
    > lang = ConveRTLanguage()
    > lang['bank']
    > lang = ConveRTLanguage()
    > lang[['bank of the river', 'money on the bank', 'bank']]
    ```
    """

    def __init__(self):
        with HiddenPrints():
            module = tfhub.load("http://models.poly-ai.com/convert/v1/model.tar.gz")
            self.model = module.signatures["encode_sequence"]

    def __getitem__(self, query: Union[str, List[str]]):
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
            tf_tensor = tf.convert_to_tensor([query])
            vec = self.model(tf_tensor)["sequence_encoding"].numpy().sum(axis=1)[0]
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])
