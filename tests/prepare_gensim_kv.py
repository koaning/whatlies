from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model = Word2Vec(common_texts, size=10, window=5, min_count=1, workers=4)
model.wv.save("tests/cache/custom_gensim_vectors.kv")
