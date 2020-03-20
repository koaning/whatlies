from whatlies.language import SpacyLanguage
from whatlies.embeddingset import EmbeddingSet
from whatlies.embedding import Embedding

lang = SpacyLanguage("en_core_web_sm")


def test_artificial_embset():
    emb = lang[['red', 'blue', 'orange']]
