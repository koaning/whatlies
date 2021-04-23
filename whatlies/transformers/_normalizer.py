from sklearn.preprocessing import normalize, FunctionTransformer

from ._transformer import SklearnTransformer


class Normalizer(SklearnTransformer):
    """
    This transformer normalizes the embeddings in an `EmbeddingSet` instance.

    Arguments:
        norm: the normalization value, could be either of `'l1'`, `'l2'` or `'max'`.
        feature: if `True` each feature (i.e. dimension) of embeddings would be normalized
            independently to unit norm; otherwise, each embedding vector would be normalized to unit norm.

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Normalizer

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Normalizer(norm='l1'))
    emb.transform(Normalizer(feature=True))
    ```
    """

    def __init__(self, norm: str = "l1", feature: bool = False, **kwargs) -> None:
        self.norm = norm
        self.feature = feature
        super().__init__(
            FunctionTransformer,
            f"norm_{norm}",
            func=lambda X: normalize(X, norm=self.norm, axis=0 if self.feature else 1),
            **kwargs,
        )
