import numpy as np
from sklearn.decomposition import PCA

from whatlies import Embedding, EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class Pca:
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

    emb.transform(Pca(3)).plot_interactive_matrix('pca_0', 'pca_1', 'pca_2')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = PCA(n_components=n_components)

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
        new_vecs = self.tfm.transform(X)
        names_out = names + [f"pca_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.pca_{self.n_components}()")
