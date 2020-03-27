import numpy as np


def embset_to_X(embset):
    names = list(embset.embeddings.keys())
    embs = np.array([i.vector for i in embset.embeddings.values()])
    return names, embs
