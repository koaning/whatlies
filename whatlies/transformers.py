from umap import UMAP
import numpy as np
from sklearn.decomposition import PCA

from whatlies import Embedding, EmbeddingSet


def embset_to_X(embset):
    names = list(embset.embeddings.keys())
    embs = np.array([i.vector for i in embset.embeddings.values()])
    return names, embs


def noise(sigma:float=0.1, seed:int=42):
    """
    This transformer adds gaussian noise to an embeddingset.
    
    Arguments:
        sigma: the amount of gaussian noise to add
        seed: seed value for random number generator

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import pca

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(noise(3))
    ```
    """
    def wrapper(embset):
        np.random.seed(seed)
        names_out, embs = embset_to_X(embset=embset)
        vectors_out = embs + np.random.normal(0, sigma, embs.shape)
        return EmbeddingSet({k: Embedding(k, v, orig=k) for k, v in zip(names_out, vectors_out)})
    return wrapper


def random_adder(n:int=2):
    """
    This transformer adds random embeddings to the embeddingset.
    
    Arguments:
        n: the number of random vectors to add
    
    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import random_added

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(random_added(3)).plot_interactive_matrix('rand_0', 'rand_1', 'rand_2')
    ```
    """
    def wrapper(embset):
        names_out, embs = embset_to_X(embset=embset)
        orig_dict = {k: Embedding(k, v, orig=k) for k, v in zip(names_out, embs)}
        new_dict = {f"rand_{k}": Embedding(f"rand_{k}", np.random.normal(0, 1, embs.shape[1])) for k in range(n)}
        return EmbeddingSet({**orig_dict, **new_dict})
    return wrapper


def pca(n_components=2, **kwargs):
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
    from whatlies.transformers import pca

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(pca(3)).plot_interactive_matrix('pca_0', 'pca_1', 'pca_2')
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

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the UMAP algorithm

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import pca

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(umap(3)).plot_interactive_matrix('umap_0', 'umap_1', 'umap_2')
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
