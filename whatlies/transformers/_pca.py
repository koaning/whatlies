from sklearn.decomposition import PCA

from whatlies.transformers import Transformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Pca(Transformer):
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
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Pca(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = PCA(n_components=n_components)

    def fit(self, embset):
        names, X = embset.to_names_X()
        self.tfm.fit(X)
        self.is_fitted = True
        return self

    def transform(self, embset):
        names, X = embset.to_names_X()
        new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.pca_{self.n_components}()")
