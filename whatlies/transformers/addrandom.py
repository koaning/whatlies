import numpy as np

from whatlies import Embedding, EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class AddRandom:
    """
    This transformer adds random embeddings to the embeddingset.

    Arguments:
        n: the number of random vectors to add

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import AddRandom

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(AddRandom(3)).plot_interactive_matrix('rand_0', 'rand_1', 'rand_2')
    ```
    """

    def __init__(self, n=1, sigma=0.1, seed=42):
        self.n = n
        self.sigma = sigma
        self.seed = seed
        self.is_fitted = False

    def __call__(self, embset):
        if not self.is_fitted:
            self.fit(embset)
        return self.transform(embset)

    def fit(self, embset):
        embset_to_X(embset=embset)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        np.random.seed(self.seed)
        orig_dict = embset.embeddings.copy()
        new_dict = {
            f"rand_{k}": Embedding(
                f"rand_{k}", np.random.normal(0, self.sigma, X.shape[1])
            )
            for k in range(self.n)
        }
        return EmbeddingSet({**orig_dict, **new_dict})
