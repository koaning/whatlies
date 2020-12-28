from typing import Union, List

import tensorflow_hub as hub
import sentencepiece as spm

from whatlies import Embedding, EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin
import tensorflow.compat.v1 as tf  # noqa: F811

tf.disable_v2_behavior()


class LiteSentenceEncoder(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a pretrained universal sentence
    encoder. This particular encoder is the [lite variant](https://tfhub.dev/google/universal-sentence-encoder-lite/2).

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
        version: version of the model to fetch

    **Usage**:

    ```python
    > from whatlies.language import LiteSentenceEncoder
    > lang = LiteSentenceEncoder()
    > lang['computer']
    > lang = LiteSentenceEncoder()
    > lang[['computer', 'human', 'dog']]
    ```
    """

    def __init__(self, version=2):
        if version not in [1, 2]:
            raise ValueError("Only model version 1 or 2 are available.")
        self.version = version

        with tf.Session() as sess:
            self.module = hub.Module(
                f"https://tfhub.dev/google/universal-sentence-encoder-lite/{version}"
            )
            spm_path = sess.run(self.module(signature="spm_path"))

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        self.encodings = self.module(
            inputs=dict(
                values=self.input_placeholder.values,
                indices=self.input_placeholder.indices,
                dense_shape=self.input_placeholder.dense_shape,
            )
        )

    def process_to_IDs_in_sparse_format(self, sentences):
        ids = [self.sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [
            [row, col] for row in range(len(ids)) for col in range(len(ids[row]))
        ]
        return values, indices, dense_shape

    def calculate_embeddings(self, messages):
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(messages)

        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(
                self.encodings,
                feed_dict={
                    self.input_placeholder.values: values,
                    self.input_placeholder.indices: indices,
                    self.input_placeholder.dense_shape: dense_shape,
                },
            )

        return message_embeddings

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**

        ```python
        > from whatlies.language import LiteSentenceEncoder
        > lang = LiteSentenceEncoder()
        > lang['computer']
        > lang = LiteSentenceEncoder()
        > lang[['computer', 'human', 'dog']]
        ```
        """
        if isinstance(query, str):
            vec = self.calculate_embeddings([query])[0]
            return Embedding(query, vec)
        vec = self.calculate_embeddings(query)
        return EmbeddingSet(*[Embedding(q, v) for q, v in zip(query, vec)])
