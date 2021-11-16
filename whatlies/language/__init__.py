from ._countvector_lang import CountVectorLanguage
from ._tfidf_lang import TFIDFVectorLanguage
from ._bpemblang import BytePairLanguage
from ._gensim_lang import GensimLanguage

from whatlies.error import NotInstalled


try:
    from ._tfhub_lang import TFHubLanguage
    from ._convert_lang import ConveRTLanguage
    from ._sentence_encode_lang import UniversalSentenceLanguage
except ModuleNotFoundError:
    TFHubLanguage = NotInstalled("TFHubLanguage", "tfhub")
    ConveRTLanguage = NotInstalled("ConveRTLanguage", "tfhub")
    UniversalSentenceLanguage = NotInstalled("UniversalSentenceLanguage", "tfhub")

try:
    from ._hftransformers_lang import HFTransformersLanguage, LaBSELanguage
except ModuleNotFoundError:
    HFTransformersLanguage = NotInstalled("HFTransformersLanguage", "transformers")
    LaBSELanguage = NotInstalled("LaBSELanguage", "transformers")

try:
    from ._sense2vec_lang import Sense2VecLanguage
except ModuleNotFoundError:
    Sense2VecLanguage = NotInstalled("Sense2VecLanguage", "sense2vec")

try:
    from ._fasttext_lang import FasttextLanguage
except ModuleNotFoundError:
    FasttextLanguage = NotInstalled("FasttextLanguage", "fasttext")

try:
    from ._floret_lang import FloretLanguage
except ModuleNotFoundError:
    FloretLanguage = NotInstalled("FloretLanguage", "floret")

try:
    from ._spacy_lang import SpacyLanguage
except ModuleNotFoundError:
    SpacyLanguage = NotInstalled("SpacyLanguage", "spacy")

try:
    from ._sentencetfm_lang import SentenceTFMLanguage
except ModuleNotFoundError:
    SentenceTFMLanguage = NotInstalled("SentenceTFMLanguage", "sentence_tfm")

try:
    from ._diet_lang import DIETLanguage
except ModuleNotFoundError:
    DIETLanguage = NotInstalled("DIETLanguage", "rasa")


__all__ = [
    "SpacyLanguage",
    "FasttextLanguage",
    "CountVectorLanguage",
    "TFIDFVectorLanguage",
    "BytePairLanguage",
    "GensimLanguage",
    "ConveRTLanguage",
    "TFHubLanguage",
    "HFTransformersLanguage",
    "UniversalSentenceLanguage",
    "SentenceTFMLanguage",
    "LaBSELanguage",
    "DIETLanguage",
    "FloretLanguage",
]
