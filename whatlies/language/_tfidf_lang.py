import warnings
from typing import Union, List, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin


class TFIDFVectorLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a tf/idf language
    backend. This object is meant for retreival, not plotting.

    This model will first train a scikit-learn
    [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    after which it will perform dimensionality reduction to make the numeric representation a vector. The reduction occurs via
    [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD),
    also from scikit-learn.

    Warning:
        This method does not implement a word embedding in the traditional sense. The interpretation needs to
        be altered. The information that is captured here only relates to the words/characters that are used in the
        text. There is no notion of meaning that should be suggested.

        Also, in order to keep this system consistent with the rest of the api you train the system when you retreive
        vectors if you just use `__getitem__`. If you want to seperate train/test you need to call `fit_manual`
        yourself or use it in a scikit-learn pipeline.

    Arguments:
        n_components: Number of components that TruncatedSVD will reduce to.
        lowercase: If the tokens need to be lowercased beforehand.
        analyzer: Which analyzer to use, can be "word", "char", "char_wb".
        ngram_range: The range that specifies how many ngrams to use.
        min_df: Ignore terms that have a document frequency strictly lower than the given threshold.
        max_df: Ignore terms that have a document frequency strictly higher than the given threshold.
        binary: Determines if the counts are binary or if they can accumulate.
        strip_accents: Remove accents and perform normalisation. Can be set to "ascii" or "unicode".
        random_state: Random state for SVD algorithm.

    For more elaborate explainers on these arguments, check out the scikit-learn
    [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer).

    **Usage**:

    ```python
    from whatlies.language import TFIDFVectorLanguage
    lang = TFIDFVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
    lang[['pizza', 'pizzas', 'firehouse', 'firehydrant']]
    ```
    """

    def __init__(
        self,
        n_components: int,
        lowercase: bool = True,
        analyzer: str = "char",
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        binary: bool = False,
        strip_accents: str = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.lowercase = lowercase
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary
        self.strip_accents = strip_accents
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.cv = TfidfVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            strip_accents=strip_accents,
        )
        self.fitted_manual = False
        self.corpus = {}

    def fit_manual(self, query):
        """
        Fit the model manually. This way you can call `__getitem__` independantly of training.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        from whatlies.language import CountVectorLanguage
        lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
        lang.fit_manual(['pizza', 'pizzas', 'firehouse', 'firehydrant'])
        lang[['piza', 'pizza', 'pizzaz', 'fyrehouse', 'firehouse', 'fyrehidrant']]
        ```
        """
        if any([len(q) == 0 for q in query]):
            raise ValueError(
                "You've passed an empty string to the language model which is not allowed."
            )
        X = self.cv.fit_transform(query)
        self.svd.fit(X)
        self.fitted_manual = True
        self.corpus = query
        return self

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a set of embeddings.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        from whatlies.language import CountVectorLanguage
        lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
        lang[['pizza', 'pizzas', 'firehouse', 'firehydrant']]
        ```
        """
        orig_str = isinstance(query, str)
        if orig_str:
            query = [query]
        if any([len(q) == 0 for q in query]):
            raise ValueError(
                "You've passed an empty string to the language model which is not allowed."
            )
        if self.fitted_manual:
            X = self.cv.transform(query)
            X_vec = self.svd.transform(X)
        else:
            X = self.cv.fit_transform(query)
            X_vec = self.svd.fit_transform(X)
        if orig_str:
            return Embedding(name=query[0], vector=X_vec[0])
        return EmbeddingSet(
            *[Embedding(name=n, vector=v) for n, v in zip(query, X_vec)]
        )

    def _prepare_queries(self, lower):
        queries = [w for w in self.corpus]
        if lower:
            queries = [w for w in queries if w.lower() == w]
        return queries

    def _calculate_distances(self, emb, queries, metric):
        vec = emb.vector
        vector_matrix = np.array([self[w].vector for w in queries])
        # there are NaNs returned, good to investigate later why that might be
        vector_matrix = np.array(
            [np.zeros(v.shape) if np.any(np.isnan(v)) else v for v in vector_matrix]
        )
        return pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)

    def score_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        metric="cosine",
        lower=False,
    ) -> List:
        """
        Retreive a list of (Embedding, score) tuples that are the most similar to the passed query.
        Note that we will only consider words that were passed in the `.fit_manual()` step.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(lower=lower)
        distances = self._calculate_distances(emb=emb, queries=queries, metric=metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        if len(self.corpus) < n:
            raise ValueError(
                f"You're trying to retreive {n} items while the corpus only trained on {len(self.corpus)}."
            )

        if len(queries) < n:
            warnings.warn(
                f"We could only find {len(queries)} feasible words. Consider changing `top_n` or `lower`",
                UserWarning,
            )

        return [(self[q], float(d)) for q, d in by_similarity[:n]]

    def embset_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        lower=False,
        metric="cosine",
    ) -> EmbeddingSet:
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most similar to the passed query.
        Note that we will only consider words that were passed in the `.fit_manual()` step.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [
            w[0] for w in self.score_similar(emb=emb, n=n, lower=lower, metric=metric)
        ]
        return EmbeddingSet({w.name: w for w in embs})
