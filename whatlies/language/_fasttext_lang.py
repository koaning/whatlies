import warnings

import numpy as np
from typing import Union, List
from sklearn.metrics import pairwise_distances

import fasttext
import fasttext.util

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet

from whatlies.language._common import SklearnTransformerMixin, HiddenPrints


class FasttextLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a fasttext language
    backend. This object is meant for retreival, not plotting.

    Important:
        The vectors are not given by this library they must be downloaded upfront.
        You can find the download links [here](https://fasttext.cc/docs/en/crawl-vectors.html).
        Note: you'll want the `bin` file, **not** the `text` file.
        To train your own fasttext model see the guide [here](https://fasttext.cc/docs/en/python-module.html#word-representation-model).

        This language backend might require you to manually install extra dependencies
        unless you installed via either;

        ```
        pip install whatlies[fasttext]
        pip install whatlies[all]
        ```


    Warning:
        You could theoretically use fasttext to train your own models with this code;

        ```
        > import fasttext
        > model = fasttext.train_unsupervised('data.txt',
                                              model='cbow',
                                              dim=10)
        > model = fasttext.train_unsupervised('data.txt',
                                              model='skipgram',
                                              dim=20,
                                              epoch=20,
                                              lr=0.1,
                                              min_count=1)
        > lang = FasttextLanguage(model)
        > lang['python']
        > model.save_model("result/data-skipgram-20.bin")
        > lang = FasttextLanguage("result/data-skipgram-20.bin")
        ```

        But you need to be aware that the fasttext library from facebook has gone stale.
        Last update on pypi was June 2019. Our preferred usecase for it is to use the pretrained vectors.
        Note that you can also import these via spaCy but this requires a packaging step.

    Arguments:
        model: name of the model to load, be sure that it's downloaded or trained beforehand

    **Usage**:

    ```python
    > from whatlies.language import FasttextLanguage
    > lang = FasttextLanguage("cc.en.300.bin")
    > lang['python']
    > lang = FasttextLanguage("cc.en.300.bin", size=10)
    > lang[['python', 'snake', 'dog']]
    ```
    """

    def __init__(self, model, size=None):
        self.size = size
        # we have to use this class to prevent the warning hidden as a print statement from the fasttext lib
        with HiddenPrints():
            if isinstance(model, str):
                self.model = fasttext.load_model(model)
            elif isinstance(model, fasttext.FastText._FastText):
                self.model = model
            else:
                raise ValueError(
                    "Language must be started with `str` or fasttext.FastText._FastText object."
                )
        if self.size:
            fasttext.util.reduce_model(self.model, self.size)

    @staticmethod
    def _input_str_legal(string):
        if sum(1 for c in string if c == "[") > 1:
            raise ValueError("only one opener `[` allowed ")
        if sum(1 for c in string if c == "]") > 1:
            raise ValueError("only one opener `]` allowed ")

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = FasttextLanguage("cc.en.300.bin")
        > lang['python']
        > lang[['python'], ['snake']]
        > lang[['nobody expects'], ['the spanish inquisition']]
        ```
        """
        if isinstance(query, str):
            self._input_str_legal(query)
            vec = self.model.get_word_vector(query)
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])

    def _prepare_queries(self, top_n, lower):
        queries = [w for w in self.model.get_words()]
        if lower:
            queries = [w for w in queries if w.is_lower]
        if top_n is not None:
            queries = queries[:top_n]
        if len(queries) == 0:
            raise ValueError(
                f"Language model has no tokens for this setting. Consider raising top_n={top_n}"
            )
        return queries

    def _calculate_distances(self, emb, queries, metric):
        vec = emb.vector
        vector_matrix = np.array([self.model.get_word_vector(w) for w in queries])
        return pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)

    def embset_proximity(
        self,
        emb: Union[str, Embedding],
        max_proximity: float = 0.1,
        top_n=20_000,
        lower=True,
        metric="cosine",
    ):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] or embeddings that are within a proximity.

        Arguments:
            emb: query to use
            max_proximity: the number of items you'd like to see returned
            top_n: likelihood limit that sets the subset of words to search
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(top_n, lower)
        distances = self._calculate_distances(emb, queries, metric)
        return EmbeddingSet(
            {w: self[w] for w, d in zip(queries, distances) if d <= max_proximity}
        )

    def embset_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        top_n=20_000,
        lower=False,
        metric="cosine",
    ):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most similar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            top_n: likelihood limit that sets the subset of words to search
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens, note that the official english model only has lower case tokens

        Important:
            This method is incredibly slow at the moment without a good `top_n` setting due to
            [this bug](https://github.com/facebookresearch/fastText/issues/1040).

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [w[0] for w in self.score_similar(emb, n, top_n, lower, metric)]
        return EmbeddingSet({w.name: w for w in embs})

    def score_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        top_n=20_000,
        lower=False,
        metric="cosine",
    ):
        """
        Retreive a list of (Embedding, score) tuples that are the most similar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            top_n: likelihood limit that sets the subset of words to search, to ignore set to `None`
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens, note that the official english model only has lower case tokens

        Important:
            This method is incredibly slow at the moment without a good `top_n` setting due
            to [this bug](https://github.com/facebookresearch/fastText/issues/1040).

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(top_n, lower)
        distances = self._calculate_distances(emb, queries, metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        if len(queries) < n:
            warnings.warn(
                f"We could only find {len(queries)} feasible words. Consider changing `top_n` or `lower`",
                UserWarning,
            )

        return [(self[q], float(d)) for q, d in by_similarity[:n]]
