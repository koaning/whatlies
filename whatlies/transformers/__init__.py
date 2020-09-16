from whatlies.transformers._transformer import Transformer

# from whatlies.transformers._pca import Pca
from whatlies.transformers._umap import Umap
from whatlies.transformers._noise import Noise
from whatlies.transformers._addrandom import AddRandom
from whatlies.transformers._tsne import Tsne
from whatlies.transformers._normalizer import Normalizer
from whatlies.error import NotInstalled


from whatlies.transformers._transformer import SklearnTransformer
from sklearn.decomposition import PCA


def Pca(n_components=2, *args, **kwargs):
    """
    Creates a PCA transformer component that you can use on an `EmbeddingSet`.

    You can pass it all the parameters from the underlying scikit-learn implementation for customisation.
    """
    return SklearnTransformer(
        PCA, name="pca", n_components=n_components, *args, **kwargs
    )


try:
    from whatlies.transformers._opentsne import OpenTsne
except ModuleNotFoundError:
    OpenTsne = NotInstalled("OpenTsne", "opentsne")

try:
    from whatlies.transformers._ivis import Ivis
except ModuleNotFoundError:
    Ivis = NotInstalled("Ivis", "ivis")

__all__ = [
    "Transformer",
    "Pca",
    "Umap",
    "Noise",
    "AddRandom",
    "Tsne",
    "OpenTsne",
    "Ivis",
    "Normalizer",
]
