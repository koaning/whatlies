from whatlies.transformers.pca import Pca
from whatlies.transformers.umap import Umap
from whatlies.transformers.noise import Noise
from whatlies.transformers.addrandom import AddRandom
from whatlies.transformers.tsne import Tsne
from whatlies.error import NotInstalled

try:
    from whatlies.transformers.opentsne import OpenTsne
except ModuleNotFoundError:
    OpenTsne = NotInstalled("OpenTsne", "opentsne")

try:
    from whatlies.transformers.ivis import Ivis
except ModuleNotFoundError:
    Ivis = NotInstalled("Ivis", "ivis")

__all__ = ["Pca", "Umap", "Noise", "AddRandom", "Tsne", "OpenTsne", "Ivis"]
