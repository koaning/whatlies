One of the goals of this package is to make it simple to quickly
benchmark the effect of embeddings. This includes embeddings that
are Non-English. In this guide we'll demonstrate how you might be
able to use this library to run simple Arabic classification benchmark
using scikit-learn and this library.

## Finding Embeddings

For Arabic there are many sources of embeddings. In this benchmark we'll
limit ourselves to [BytePair](https://nlp.h-its.org/bpemb/ar/)
embeddings and pretrained [BERT](https://github.com/alisafaya/Arabic-BERT).

There are also plenty of other embeddings out there. There's gensim
embeddings available via [AraVec](https://github.com/bakrianoo/aravec) and
there's also pretrained subword-embeddings in [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html#models).
There's also support for the Egyptian dialect but we'll ignore these for now.

## The task

The task we'll benchmark is sentiment analysis for
[arabic tweets](https://www.kaggle.com/mksaad/arabic-sentiment-twitter-corpus). This benchmark
was part of discussion on [github](https://github.com/RasaHQ/whatlies/issues/262).

```python
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from whatlies.language import BytePairLanguage, HFTransformersLanguage
from whatlies.transformers import Umap

lang_bp1 = BytePairLanguage("ar", vs=10000, dim=300)
lang_bp2 = BytePairLanguage("ar", vs=200000, dim=300)
lang_hf = HFTransformersLanguage("asafaya/bert-base-arabic")

df = pd.concat([
    pd.read_csv("test_Arabic_tweets_negative_20190413.tsv", sep="\t", names=["label", "text"]),
    pd.read_csv("test_Arabic_tweets_positive_20190413.tsv", sep="\t", names=["label", "text"])
], axis=0).sample(frac=1).reset_index(drop=True)

# Sample a small list such that the interactive charts render swiftly.
small_text_list = list(set(df[:1000]['text']))

def mk_plot(lang, title=""):
    return (lang[small_text_list]
            .transform(Umap(2))
            .plot_interactive(annot=False)
            .properties(title=title, width=200, height=200))

mk_plot(lang_bp1, "bp_small") | mk_plot(lang_bp2, "bp_big") | mk_plot(lang_hf, "huggingface")
```
