from .spacy_lang import SpacyLanguage
from .sense2vec_lang import Sense2VecLanguage
from .fasttext_lang import FasttextLanguage
from .countvector_lang import CountVectorLanguage
from .bpemblang import BytePairLanguage
from .bpemblang import BytePairLanguage as BytePairLang
from .gensim_lang import GensimLanguage

from whatlies.error import NotInstalled

try:
    from .convert_lang import ConveRTLanguage
    from .tfhub_lang import TFHubLanguage
except ModuleNotFoundError as e:
    TFHubLanguage = NotInstalled("TFHubLanguage", "tfhub")
    ConveRTLanguage = NotInstalled("ConveRTLanguage", "tfhub")

try:
    from .hftransformers_lang import HFTransformersLanguage
except ModuleNotFoundError as e:
    HFTransformersLanguage = NotInstalled("HFTransformersLanguage", "transformers")


__all__ = [
    "SpacyLanguage",
    "Sense2VecLanguage",
    "FasttextLanguage",
    "CountVectorLanguage",
    "BytePairLang",
    "BytePairLanguage",
    "GensimLanguage",
    "ConveRTLanguage",
    "TFHubLanguage",
    "HFTransformersLanguage",
]
