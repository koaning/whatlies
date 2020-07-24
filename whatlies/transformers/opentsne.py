import numpy as np
from openTSNE import TSNE

from whatlies import EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class OpenTsne:
    """
    This transformer transformers all vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of tsne. This implementation used
    [open-tsne](https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html).

    Important:
        OpenTSNE is a faster variant of TSNE but it only allows for <2 components.
        You may also notice that it is relatively slow. This unfortunately is a fact of life.

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the OpenTsne implementation, includes things like `perplexity` [link](https://opentsne.readthedocs.io/en/latest/api/index.html)

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import OpenTsne

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(OpenTsne(2)).plot_interactive_matrix('tsne_0', 'tsne_1')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = TSNE(n_components=n_components, **kwargs)

    def __call__(self, embset):
        if not self.is_fitted:
            self.fit(embset)
            self.is_fitted = True
        return self.transform(embset)

    def fit(self, embset):
        names, X = embset_to_X(embset=embset)
        self.emb = self.tfm.fit(X)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        new_vecs = np.array(self.emb.transform(X))
        names_out = names + [f"tsne_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.tsne_{self.n_components}()")
