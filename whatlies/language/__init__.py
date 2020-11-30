from ._spacy_lang import SpacyLanguage
from ._fasttext_lang import FasttextLanguage
from ._countvector_lang import CountVectorLanguage
from ._tfidf_lang import TFIDFVectorLanguage
from ._bpemblang import BytePairLanguage
from ._gensim_lang import GensimLanguage

from whatlies.error import NotInstalled


try:
    from ._tfhub_lang import TFHubLanguage
    from ._convert_lang import ConveRTLanguage
    from ._sentence_encode_lang import UniversalSentenceLanguage
except ModuleNotFoundError as e:
    TFHubLanguage = NotInstalled("TFHubLanguage", "tfhub")
    ConveRTLanguage = NotInstalled("ConveRTLanguage", "tfhub")
    UniversalSentenceLanguage = NotInstalled("UniversalSentenceLanguage", "tfhub")

try:
    from ._hftransformers_lang import HFTransformersLanguage
except ModuleNotFoundError as e:
    HFTransformersLanguage = NotInstalled("HFTransformersLanguage", "transformers")

try:
    from ._sense2vec_lang import Sense2VecLanguage
except ModuleNotFoundError as e:
    Sense2VecLanguage = NotInstalled("Sense2VecLanguage", "sense2vec")

try:
    from ._sentencetfm_lang import SentenceTFMLanguage
except ModuleNotFoundError as e:
    SentenceTFMLanguage = NotInstalled("SentenceTFMLanguage", "sentence_tfm")


__all__ = [
    "SpacyLanguage",
    "Sense2VecLanguage",
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
]
