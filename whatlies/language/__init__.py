from ._spacy_lang import SpacyLanguage
from ._fasttext_lang import FasttextLanguage
from ._countvector_lang import CountVectorLanguage
from ._bpemblang import BytePairLanguage
from ._gensim_lang import GensimLanguage
from ._convert_lang import ConveRTLanguage

from whatlies.error import NotInstalled

try:
    from ._tfhub_lang import TFHubLanguage
except ModuleNotFoundError as e:
    TFHubLanguage = NotInstalled("TFHubLanguage", "tfhub")

try:
    from ._sentence_encode_lang import UniversalSentenceLanguage
except ModuleNotFoundError as e:
    UniversalSentenceLanguage = NotInstalled("UniversalSentenceLanguage", "tfhub")

try:
    from ._hftransformers_lang import HFTransformersLanguage
except ModuleNotFoundError as e:
    HFTransformersLanguage = NotInstalled("HFTransformersLanguage", "transformers")

try:
    from ._sense2vec_lang import Sense2VecLanguage
except ModuleNotFoundError as e:
    Sense2VecLanguage = NotInstalled("Sense2VecLanguage", "sense2vec")


__all__ = [
    "SpacyLanguage",
    "Sense2VecLanguage",
    "FasttextLanguage",
    "CountVectorLanguage",
    "BytePairLanguage",
    "GensimLanguage",
    "ConveRTLanguage",
    "TFHubLanguage",
    "HFTransformersLanguage",
    "UniversalSentenceLanguage",
]
