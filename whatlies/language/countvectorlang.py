import warnings

import numpy as np
from typing import Union, List
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet

from whatlies.language.common import SklearnTransformerMixin


class CountVectorLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a countvector language
    backend. This object is meant for retreival, not plotting.

    This model will first train a scikit-learn
    [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer)
    after it will perform dimensionality reduction to make the numeric representation a vector. The reduction occurs via
    [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD),
    also from scikit-learn.

    Warning:
        To keep this system consistent with the rest of the api you train the system when you retreive vectors.

    Arguments:
        model: name of the model to load, be sure that it's downloaded or trained beforehand

    **Usage**:

    ```python
    > from whatlies.language import CountVectorLanguage
    > lang = CountVectorLanguage(size=20, ngram_range=(1, 2), analyzer="char")
    > lang[['pizza', 'pizzas', 'pie', 'firehydrant']]
    ```
    """

    def __init__(
        self,
        n_components,
        lowercase=True,
        analyzer="char",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        binary=False,
        strip_accents=None,
    ):
        self.svd = TruncatedSVD(n_components=n_components)
        self.cv = CountVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            strip_accents=strip_accents,
        )

    @staticmethod
    def _input_str_legal(string):
        if sum(1 for c in string if c == "[") > 1:
            raise ValueError("only one opener `[` allowed ")
        if sum(1 for c in string if c == "]") > 1:
            raise ValueError("only one opener `]` allowed ")

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a set of embeddings.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        > from whatlies.language import CountVectorLanguage
        > lang = CountVectorLanguage(size=20, ngram_range=(1, 2), analyzer="char")
        > lang[['pizza', 'pizzas', 'pie', 'firehydrant']]
        ```
        """
        X = self.cv.fit_transform(query)
        X_vec = self.svd.fit_transform(X)
        return EmbeddingSet(*[Embedding(name=n, vector=v) for n, v in zip(query, X_vec)])

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
