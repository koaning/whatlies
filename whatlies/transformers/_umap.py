import warnings

from umap import UMAP
from numba import NumbaPerformanceWarning

from whatlies.transformers import Transformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Umap(Transformer):
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

    emb.transform(Umap(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = UMAP(n_components=n_components, **kwargs)

    def fit(self, embset):
        names, X = embset.to_names_X()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            self.tfm.fit(X)
        self.is_fitted = True
        return self

    def transform(self, embset):
        names, X = embset.to_names_X()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.umap_{self.n_components}()")
