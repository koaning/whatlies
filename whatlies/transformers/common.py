from copy import deepcopy

from whatlies import Embedding


def new_embedding_dict(names_new, vectors_new, old_embset):
    new_embeddings = {}
    for k, v in zip(names_new, vectors_new):
        new_emb = (
            deepcopy(old_embset[k])
            if k in old_embset.embeddings.keys()
            else Embedding(k, v, orig=k)
        )
        new_emb.name = k
        new_emb.vector = v
        new_embeddings[k] = new_emb
    return new_embeddings
