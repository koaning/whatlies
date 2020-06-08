import pytest
import numpy as np

from whatlies.language import CountVectorLanguage


def test_basic_usage():
    lang = CountVectorLanguage(n_components=20, ngram_range=(1, 2), analyzer="char")
