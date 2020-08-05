import pytest
import numpy as np

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import SpacyLanguage


@pytest.fixture()
def color_lang():
    vector_data = {
        "red": np.array([1.0, 0.0]),
        "green": np.array([0.5, 0.5]),
        "blue": np.array([0.0, 1.0]),
        "purple": np.array([0.0, 1.0]),
    }

    vocab = Vocab(strings=vector_data.keys())
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


def test_basic_usage(color_lang):
    queries = [
        "green is blue and yellow",
        "purple is red and blue",
        "purple isn't same as red!",
        "[red and blue] is a like a glue!",
    ]
    emb = color_lang[queries]
    assert len(emb) == 4
    assert emb[queries[0]].name == "green is blue and yellow"
    assert emb[queries[0]].vector.shape == (2,)


@pytest.mark.parametrize(
    "query",
    [
        "no blue [closing bracket",
        "no red opening] bracket",
        "[more] than one blue [context]",
    ],
)
def test_invalid_query_raises_error(color_lang, query):
    with pytest.raises(ValueError, match="bracket"):
        color_lang[query]


@pytest.mark.parametrize(
    "query, context, context_pos",
    [
        ("I'm going [blue]", "blue", (2, 3)),
        ("you should pre-order your [red shirt]", "red shirt", (6, 8)),
        ("Me: today is the [best green day] of my life", "best green day", (5, 8)),
        ("[red and blue] is like a glue!", "red and blue", (0, 3)),
        (
            "Let's have Fun without a blue context.",
            "Let's have Fun without a blue context.",
            (0, None),
        ),
    ],
)
def test_embedding_is_correct(color_lang, query, context, context_pos):
    emb = color_lang[query]
    assert emb.vector.shape == (2,)
    assert emb.name == query

    clean_query = query.replace("[", "").replace("]", "")
    doc = color_lang.model(clean_query)
    span = doc[context_pos[0] : context_pos[1]]
    assert str(span) == context
    assert np.allclose(emb.vector, span.vector)


def test_score_similar_one(color_lang):
    scores = color_lang.score_similar("blue", n=2, prob_limit=None, lower=False)
    print(scores)
    assert all([s[1] == 0 for s in scores])
    assert "blue" in [s[0].name for s in scores]
    assert "purple" in [s[0].name for s in scores]


@pytest.mark.parametrize(
    "string, array",
    zip(
        ["red", "red green [blue] purple", "green [red blue] pink"],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])],
    ),
)
def test_lang_retreival(color_lang, string, array):
    assert np.isclose(color_lang[string].vector, array).all()


def test_single_token_words(color_lang):
    # test for issue here: https://github.com/RasaHQ/whatlies/issues/5
    assert len(color_lang["red"].vector) > 0


def test_raise_warning(color_lang):
    print([w for w in color_lang.nlp.vocab])
    with pytest.warns(UserWarning):
        color_lang.score_similar("red", 100, prob_limit=None, lower=False)
