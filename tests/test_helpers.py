from whatlies import Embedding, EmbeddingSet
from whatlies.helpers import reverse_strings


def test_reverse_strings():
    embset = EmbeddingSet(Embedding(name="helloworld", vector=[1, 2])).pipe(
        reverse_strings
    )
    emb = [e for e in embset][0]
    assert emb.name == "dlrowolleh"
