import warnings

import numpy as np
from typing import Union, List, Tuple
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
    after which it will perform dimensionality reduction to make the numeric representation a vector. The reduction occurs via
    [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD),
    also from scikit-learn.

    Warning:
        This method does not implement a word embedding in the traditional sense. The interpretation needs to
        be altered. The information that is captured here only relates to the words/characters that are used in the
        text. There is no notion of meaning that should be suggested.

        Also, in order to keep this system consistent with the rest of the api you train the system when you retreive
        vectors

    Arguments:
        n_components: Number of components that TruncatedSVD will reduce to.
        lowercase: If the tokens need to be lowercased beforehand.
        analyzer: Which analyzer to use, can be "word", "char", "char_wb".
        ngram_range: The range that specifies how many ngrams to use.
        min_df: Ignore terms that have a document frequency strictly lower than the given threshold.
        max_df: Ignore terms that have a document frequency strictly higher than the given threshold.
        binary: Determines if the counts are binary or if they can accumulate.
        strip_accents: Remove accents and perform normalisation. Can be set to "ascii" or "unicode".

    For more elaborate explainers on these arguments, check out the scikit-learn
    [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer).

    **Usage**:

    ```python
    > from whatlies.language import CountVectorLanguage
    > lang = CountVectorLanguage(n_components=20, ngram_range=(1, 2), analyzer="char")
    > lang[['pizza', 'pizzas', 'pie', 'firehydrant']]
    ```
    """

    def __init__(
        self,
        n_components: int,
        lowercase: bool = True,
        analyzer: str = "char",
        ngram_range: Tuple[int] = (1, 1),
        max_df: Union[int, float] = 1.0,
        min_df: Union[int, float] = 1,
        binary: bool = False,
        strip_accents: str = None,
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

    def __getitem__(self, query: List[str]):
        """
        Retreive a set of embeddings.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        > from whatlies.language import CountVectorLanguage
        > lang = CountVectorLanguage(n_components=20, ngram_range=(1, 2), analyzer="char")
        > lang[['pizza', 'pizzas', 'pie', 'firehydrant']]
        ```
        """
        X = self.cv.fit_transform(query)
        X_vec = self.svd.fit_transform(X)
        return EmbeddingSet(*[Embedding(name=n, vector=v) for n, v in zip(query, X_vec)])
