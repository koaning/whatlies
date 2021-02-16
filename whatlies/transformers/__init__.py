from whatlies.transformers._pca import Pca

from whatlies.transformers._noise import Noise
from whatlies.transformers._addrandom import AddRandom
from whatlies.transformers._tsne import Tsne
from whatlies.transformers._normalizer import Normalizer
from whatlies.transformers._transformer import SklearnTransformer, Transformer
from whatlies.error import NotInstalled

try:
    from whatlies.transformers._umap import Umap
except ImportError:
    Umap = NotInstalled("Umap", "umap")


__all__ = [
    "SklearnTransformer",
    "Transformer",
    "Pca",
    "Umap",
    "Tsne",
    "Noise",
    "AddRandom",
    "Normalizer",
]
