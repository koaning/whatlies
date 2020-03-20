import pytest
import numpy as np
from whatlies.language import SpacyLanguage, _selected_idx_spacy


lang = SpacyLanguage("en_core_web_sm")


def test_single_token_words():
    # test for https://github.com/RasaHQ/whatlies/issues/5
    assert np.sum(lang["red"].vector) > 0


@pytest.mark.parametrize(
    "triplets",
    zip(
        ["red", "red green", "red green [blue] purple", "red [green blue] pink"],
        [0, 0, 2, 1],
        [1, 2, 3, 3],
    ),
)
def test_select_idx_func(triplets):
    string, start, end = triplets
    assert _selected_idx_spacy(string) == (start, end)
