from typing import Union, List, Tuple

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
    > from whatlies.language import CountVectorLanguage
    > lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
    > lang[['pizza', 'pizzas', 'firehouse', 'firehydrant']]
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
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.cv = CountVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            strip_accents=strip_accents,
        )
        self.fitted_manual = False

    def fit_manual(self, query):
        """
        Fit the model manually. This way you can call `__getitem__` independantly of training.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        > from whatlies.language import CountVectorLanguage
        > lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
        > lang.fit_manual(['pizza', 'pizzas', 'firehouse', 'firehydrant'])
        > lang[['piza', 'pizza', 'pizzaz', 'fyrehouse', 'firehouse', 'fyrehidrant']]
        ```
        """
        X = self.cv.fit_transform(query)
        self.svd.fit(X)
        self.fitted_manual = True
        return self

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a set of embeddings.

        Arguments:
            query: list of strings

        **Usage**

        ```python
        > from whatlies.language import CountVectorLanguage
        > lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
        > lang[['pizza', 'pizzas', 'firehouse', 'firehydrant']]
        ```
        """
        orig_str = isinstance(query, str)
        if orig_str:
            query = list(query)
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
