from umap import UMAP
import numpy as np
from sklearn.decomposition import PCA

from whatlies import Embedding, EmbeddingSet


def embset_to_X(embset):
    names = list(embset.embeddings.keys())
    embs = np.array([i.vector for i in embset.embeddings.values()])
    return names, embs


def noise(sigma=0.1, seed=42):
    """
    This transformer adds gaussian noise to an embeddingset.

    Usage:

    ```python
    from whatlies.transformers import noise
    embset.transform(noise(2))
    ```
    """
    def wrapper(embset):
        np.random.seed(seed)
        names_out, embs = embset_to_X(embset=embset)
        vectors_out = embs + np.random.normal(0, sigma, embs.shape)
        return EmbeddingSet({k: Embedding(k, v, orig=k) for k, v in zip(names_out, vectors_out)})
    return wrapper


def pca(n_components=2, **kwargs):
    """
    This transformer scales all the vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of principal component analysis. We're using the implementation found in
    [scikit-learn]](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

    Usage:

    ```python
    from whatlies.transformers import pca
    embset.transform(pca(2))
    ```
    """
    def wrapper(embset):
        tfm = PCA(n_components=n_components, **kwargs)
        names, embs = embset_to_X(embset=embset)
        new_vecs = tfm.fit_transform(embs)
        names_out = names + [f'pca_{i}' for i in range(n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(n_components)])
        return EmbeddingSet({k: Embedding(k, v, orig=k) for k, v in zip(names_out, vectors_out)},
                            name=f"{embset.name}.pca_{n_components}()")
    return wrapper


def umap(n_components=2, **kwargs):
    """
    This transformer transformers all vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of umap. We're using the implementation in [umap-learn](https://umap-learn.readthedocs.io/en/latest/).

    Usage:

    ```python
    from whatlies.transformers import umap
    embset.transform(umap(2))
    ```
    """
    def wrapper(embset):
        tfm = UMAP(n_components=n_components, **kwargs)
        names, embs = embset_to_X(embset=embset)
        new_vecs = tfm.fit_transform(embs)
        names_out = names + [f'umap_{i}' for i in range(n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(n_components)])
        return EmbeddingSet({k: Embedding(k, v, orig=k) for k, v in zip(names_out, vectors_out)},
                            name=f"{embset.name}.umap_{n_components}")
    return wrapper
