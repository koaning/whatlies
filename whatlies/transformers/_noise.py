import numpy as np
from sklearn.preprocessing import FunctionTransformer

from whatlies.transformers import Transformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Noise(Transformer):
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
        super().__init__()
        self.seed = seed
        self.tfm = FunctionTransformer(
            lambda X: X + np.random.normal(0, sigma, X.shape)
        )

    def fit(self, embset):
        self.is_fitted = True
        return self

    def transform(self, embset):
        names, X = embset.to_names_X()
        np.random.seed(self.seed)
        new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(
            new_dict,
            name=f"{embset.name}",
        )
