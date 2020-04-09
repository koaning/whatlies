import warnings

from umap import UMAP
import numpy as np
from numba import NumbaPerformanceWarning

from whatlies import Embedding, EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class Umap:
    """
    This transformer transformers all vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of umap. We're using the implementation in [umap-learn](https://umap-learn.readthedocs.io/en/latest/).

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the UMAP algorithm

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Umap

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Umap(3)).plot_interactive_matrix('umap_0', 'umap_1', 'umap_2')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = UMAP(n_components=n_components, **kwargs)

    def __call__(self, embset):
        if not self.is_fitted:
            self.fit(embset)
        return self.transform(embset)

    def fit(self, embset):
        names, X = embset_to_X(embset=embset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            self.tfm.fit(X)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            new_vecs = self.tfm.transform(X)
        names_out = names + [f"umap_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.umap_{self.n_components}()")
