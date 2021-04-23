import numpy as np
from sklearn.preprocessing import FunctionTransformer

from ._transformer import SklearnTransformer


class Noise(SklearnTransformer):
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
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Noise(3))
    ```
    """

    def __init__(self, sigma=0.1, seed=42, **kwargs):
        np.random.seed(seed)
        super().__init__(
            FunctionTransformer,
            f"noise_{sigma}",
            func=lambda X: X + np.random.normal(0, sigma, X.shape),
            **kwargs,
        )
