from sklearn.decomposition import PCA

from ._transformer import SklearnTransformer


class Pca(SklearnTransformer):
    """
    This transformer scales all the vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of principal component analysis. We're using the implementation found in
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the PCA from scikit-learn

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Pca

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Pca(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components, **kwargs):
        super().__init__(
            PCA, f"pca_{n_components}", n_components=n_components, **kwargs
        )
