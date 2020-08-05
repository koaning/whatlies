from .spacy_lang import SpacyLanguage
from .sense2vec_lang import Sense2VecLanguage
from .fasttext_lang import FasttextLanguage
from .countvector_lang import CountVectorLanguage
from .bpemblang import BytePairLanguage
from .bpemblang import BytePairLanguage as BytePairLang
from .convert_lang import ConveRTLanguage
from .gensim_lang import GensimLanguage
from .tfhub_lang import TFHubLanguage
from .hftransformers_lang import HFTransformersLanguage


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
