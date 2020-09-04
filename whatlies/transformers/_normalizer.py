from sklearn.preprocessing import normalize

from whatlies.transformers import Transformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Normalizer(Transformer):
    """
    This transformer normalizes the embeddings in an `EmbeddingSet` instance.

    Arguments:
        norm: the normalization value, could be either of `'l1'`, `'l2'` or `'max'`.
        feature: if `True` each feature (i.e. dimension) of embeddings would be normalized
            independently to unit norm; otheriwse, each embedding vector would be normazlied to unit norm.

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Normalizer

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Normalizer(norm='l1'))
    emb.transform(Normalizer(feature=True))
    ```
    """

    def __init__(self, norm: str = "l1", feature: bool = False) -> None:
        super().__init__()
        self.norm = norm
        self.feature = feature

    def fit(self, embset: EmbeddingSet) -> "Normalizer":
        self.is_fitted = True
        return self

    def transform(self, embset: EmbeddingSet) -> EmbeddingSet:
        names, X = embset.to_names_X()
        axis = 0 if self.feature else 1
        X = normalize(X, norm=self.norm, axis=axis)
        new_dict = new_embedding_dict(names, X, embset)
        return EmbeddingSet(new_dict, name=embset.name)
