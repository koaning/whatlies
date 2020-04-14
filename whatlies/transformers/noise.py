import numpy as np
from sklearn.preprocessing import FunctionTransformer

from whatlies import EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class Noise:
    """
    This transformer adds gaussian noise to an embeddingset.

    Arguments:
        sigma: the amount of gaussian noise to add
        seed: seed value for random number generator

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Noise

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Noise(3))
    ```
    """

    def __init__(self, sigma=0.1, seed=42):
        self.is_fitted = False
        self.seed = seed
        self.tfm = FunctionTransformer(
            lambda X: X + np.random.normal(0, sigma, X.shape)
        )

    def __call__(self, embset):
        if not self.is_fitted:
            self.fit(embset)
        return self.transform(embset)

    def fit(self, embset):
        names, X = embset_to_X(embset=embset)
        self.tfm.fit(X)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        np.random.seed(self.seed)
        new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}",)
