from umap import UMAP

from whatlies.transformers._transformer import SklearnTransformer


class Umap(SklearnTransformer):
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
        super().__init__(
            UMAP, f"umap_{n_components}", n_components=n_components, **kwargs
        )
