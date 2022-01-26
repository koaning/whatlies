<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>

One of the goals of this package is to make it simple to explore embeddings.
This includes embeddings that are Non-English. In this guide we'll demonstrate how you might be
able to use this library to run simple Arabic classification benchmark
using scikit-learn and this library. This benchmark was part of discussion
on [github](https://github.com/koaning/whatlies/issues/262).

If you want to follow along, make sure that this tool is installed.

```
pip install whatlies
```

If you'd like to also try out the heavy transformer model, you'll also
want to install extra dependencies.

```
pip install "whatlies[transformers]"
```

## Finding Embeddings

For Arabic there are many sources of embeddings. In this benchmark we'll
limit ourselves to [BytePair](https://nlp.h-its.org/bpemb/ar/)
embeddings and pretrained [BERT](https://github.com/alisafaya/Arabic-BERT).

There are also plenty of other embeddings out there. There's gensim
embeddings available via [AraVec](https://github.com/bakrianoo/aravec) and
there's also pretrained subword-embeddings in [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html#models).
There's also support for the Egyptian dialect but we'll ignore these for now.

## The Task

The task we'll benchmark is sentiment analysis for
[arabic tweets](https://www.kaggle.com/mksaad/arabic-sentiment-twitter-corpus). It's a simple classification problem with
two classes; positive and negative. We'll assume the labels for the dataset are accurate but we should
remind ourselves to check correctness later if we intend to use this dataset for a real life use-case. For
more info on this topic [watch here](https://www.youtube.com/watch?v=Czto6GzJah8&list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb&index=32&ab_channel=Rasa).

## Explore

Before you run the big benchmark, it makes sense to explore the embeddings first.

Let's start by loading them in.

```python
from whatlies.language import BytePairLanguage, HFTransformersLanguage

lang_bp1 = BytePairLanguage("ar", vs=10000, dim=300)
lang_bp2 = BytePairLanguage("ar", vs=200000, dim=300)

# Feel free to remove `lang_hf` from the benchmark if you want want quick results.
# These BERT-style embeddings are very compute heavy and can take a while to benchmark.
lang_hf = HFTransformersLanguage("asafaya/bert-base-arabic")
```

A popular method of visualising embeddings is to make a scatterplot of clusters. We'll
look at a few text examples by embedding them and reducing their dimensionality via
Umap.

```python
import pandas as pd

from whatlies.transformers import Umap

# Read in the dataframes from Kaggle
df = pd.concat([
    pd.read_csv("test_Arabic_tweets_negative_20190413.tsv", sep="\t"),
    pd.read_csv("test_Arabic_tweets_positive_20190413.tsv", sep="\t")
], axis=0).sample(frac=1).reset_index(drop=True)
df.columns = ["label", "text"]

# Next we clean the dataset
df = (df
  .loc[lambda d: d['text'].str.len() < 200]
  .drop_duplicates()
  .sample(frac=1)
  .reset_index(drop=True))

# Sample a small list such that the interactive charts render swiftly.
small_text_list = list(set(df[:1000]['text']))

def mk_plot(lang, title=""):
    return (lang[small_text_list]
            .transform(Umap(2))
            .plot_interactive(annot=False)
            .properties(title=title, width=200, height=200))

mk_plot(lang_bp2, "bp_big") | mk_plot(lang_hf, "huggingface")
```

The results of this code are viewable below. Note that these charts are fully interactive
and they'll show the text containing the tweets when you hover over them.

<div id="vis1"></div>

<script>
fetch('charts.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis1', out);
})
.catch(err => { throw err });
</script>

You might recognize some clusters, which is nice, but it's no benchmark yet.

## Benchmark

The code below will actually run the actual benchmark. What's important to note is that
we'll train on 7829 examples and we'll test on a holdout set of 1000 examples. In the
interest of time we won't do a k-fold validation. After all, the main goal is to show
how you might quickly prototype a solution using whatlies.

```python
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from memo import grid, memfile, time_taken

# Split dataframe into a train/test set.
X_train, X_test, y_train, y_test = train_test_split(
    list(df['text']),
    df['label'],
    test_size=1000
)

# Create a dictionary with all of our embedding settings.
# Note that we've removed lang_hf in order to make it easy to run on a laptop.
embedders = {
    'nada': 'drop',
    'bp1': lang_bp1,
    'bp2': lang_bp2,
    # 'hf': lang_hf
}

@memfile("arabic-sentences-benchmark.jsonl")
@time_taken()
def run_experiment(embedder="nada", train_size=1000, smooth=1, ngram=True):
    # This featurizers step is a list that is used in a scikit-learn pipeline.
    featurizers = [
        ('cv-base', CountVectorizer()),
        ('cv-ngram', CountVectorizer(ngram_range=(2, 4)) if ngram else 'drop'),
        ('emb', embedders[embedder])
    ]

    # After featurization we apply a Logistic Regression
    pipe = Pipeline([
        ("feat", FeatureUnion(featurizers)),
        ("mod", LogisticRegression(C=smooth, max_iter=500))
    ])

    # The trained pipeline is used to make a prediction.
    y_pred = pipe.fit(X_train[:train_size], y_train[:train_size]).predict(X_test)

    # By returning a dictionary `memo` will be able to properly log this.
    return {"valid_accuracy": float(np.mean(y_test == y_pred)),
            "train": float(np.mean(y_train == pipe.predict(X_train)))}

# The grid will loop over all the options and generate a progress bar
# that means it's easy to run from the command line in the background.
for setting in grid(embedder=['nada', 'bp1', 'bp2', 'hf'],
                    smooth=[1, 0.1],
                    ngram=[True, False],
                    train_size=[100, 250, 500, 1000, 2000,
                                3000, 4000, 5000, 6000, 7000]):
    run_experiment(**setting)
    print(setting)
```

There's a few things to observe.

1. We're using [memo](https://koaning.github.io/memo/) to handle the logging. The `memfile` decorator grabs
app the keyword arguments and logs it together with the dictionary output.
2. We're first embedding the text numerically, after which we apply a logistic regression. A logistic regression
won't give us an upper limit of what we might be able to achieve after fine-tuning but it should serve
as a reasonable lower bound of what we could expect.
3. Besides testing the effect of the embedding we'll also have a look at the effect of training set size (`train_size`),
parameter smoothing (`smooth`) on the logistic regression and the effect of adding subword countvectors (`ngram`).
4. The benchmark will take a while! We've turned off the huggingface model so that you can get quick results
 locally but we will show our results below.

## Results

You can play with the full benchmark dataset by exploring the parallel coordinates chart
in a seperate tab [here](hiplot-results.html) but we'll focus on the most important
observations below.

## Overview

It seems that smoothing and adding countvectors for the subwords doesn't contribute
the most substantial predictive power. They can be tuned to add that 1% extra boost
but the most substaintial gain in validation accuracy is contributed by a large
training set and by chosing the huggingface embeddings.

Feel free to play around with the charts below, they are fully interactive.

<div id="vis2"></div>

<script>
fetch('details-1.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis2', out);
})
.catch(err => { throw err });
</script>

The huggingface embeddings do come with a downside however.

<div id="vis3"></div>

<script>
fetch('details-2.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#vis3', out);
})
.catch(err => { throw err });
</script>

The huggingface embeddings are *much* slower. Notice the log-scale on the y-axis and note how
the there's a big upfront calculation cost even for only 100 training examples. These
results ran on a modern server and it can take up to 8 minutes to train/process only 8000
datapoints. This pales in comparison to the other approaches. That said, we're rapid prototyping
here so we should keep in mind that we might be able to the compute time down if we build directly
on top of hugginface.

We also see that the train performance keeps increasing while the test performance
starts to flatten. If we were to increase the train set further we might need to keep
beware of overfitting.

## Conclusion

Maybe the huggingface results look promising, maybe the compute time is a concern for you or
maybe you've got a use-case where you need even more predictive power. Either way, we hope you
see the potential for this library when it comes to rapid prototyping. The goal here wasn't to
be "state of the art". Rather; whatlies allows you to try out a whole bunch of embeddings relatively
quickly so you don't need to worry about integrating with all sorts of embedding backends when
you're just starting out. This benchmark alone should not be considered as "enough" evidence to
put a model into production, but it might be enough evidence to continue iterating.

If you're curious about extending this benchmark; feel absolutely free! You might want to try
out the `LaBSELanguage`. It's a multi-language model that should support 100+ languages. We'd also
love to hear if there's useful tricks we're missing out on. We're especially keen to hear if there's
tools missing for Non-English language.

If you're interested in running this notebook yourself, you can download it [here](arabic-tweets-exercise.ipynb).
