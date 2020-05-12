<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>


In this tool we have support for different language backends and
depending on the language backend you may get slightly different behavior.

## Multiple Tokens

We can have spaCy summerize multiple tokens if we'd like.

```python
from whatlies.language.language import SpacyLanguage
from whatlies.transformers import Pca

lang = SpacyLanguage("en_core_web_sm")

contexts = ["i am super duper happy",
            "happy happy joy joy",
            "programming is super fun!",
            "i am going crazy i hate it",
            "boo and hiss",]

emb = lang[contexts]
emb.transform(Pca(2)).plot_interactive('pca_0', 'pca_1')
```

<div id="c1"></div>

<script>
fetch('spacyvec-1.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#c1', out);
})
.catch(err => { throw err });
</script>

Under the hood it will be calculating the averages of the
embeddings but we can still plot these.


## Bert Style

But spaCy also offers transformers these days, which means that
we can play with a extra bit of context.

```bash
pip install spacy-transformers
python -m spacy download en_trf_robertabase_lg
```

With these installed we can now use the same spaCy language
backend to play with transformers. Here's an example of
two embeddings selected with context.

```python
lang = SpacyLanguage("en_trf_robertabase_lg")

np.array_equal(lang['Going to the [store]'].vector,
               lang['[store] this in the drawer please.'].vector)  # False
```

In the first case we get the embedding for `store` in the context of
`Going to the store` while in the second case we have `store` in the
context of `store this in the drawer please`.

## Sense to Vec

We also have support for the [sense2vec model](https://github.com/explosion/sense2vec). To
get it to work you first need to download and unzip the pretrained vectors
found [here](https://github.com/explosion/sense2vec#pretrained-vectors) but after
that you should be able to retreive tokens with context. They way you fetch these
tokens is a bit ... different though.

```python
from whatlies.language.language import Sense2VecLanguage
from whatlies.transformers import Pca

lang = Sense2VecLanguage("path/downloaded/s2v")

words = ["bank|NOUN", "bank|VERB", "duck|NOUN", "duck|VERB",
         "dog|NOUN", "cat|NOUN", "jump|VERB", "run|VERB",
         "chicken|NOUN", "puppy|NOUN", "kitten|NOUN", "carrot|NOUN"]
emb = lang[words]
```
From here one we're back to normal embeddingsets though. So we can
plot whatever we feel like.

```python
p1 = emb.plot_interactive("dog|NOUN", "jump|VERB")
p2 = emb.transform(Pca(2)).plot_interactive("pca_0", "pca_1")
p1 | p2
```

<div id="s1"></div>

<script>
fetch('sense2vec-1.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#s1', out);
})
.catch(err => { throw err });
</script>


Notice how `duck|VERB` is certainly different from `duck|NOUN`.

### Similarity

Another nice feature of `sense2vec` is the ability to find
tokens that are nearby. We could do the following.

```python
lang.score_similar("duck|VERB")
```

This will result in a long list with embedding-score tuples.

```
[(Emb[crouch|VERB], 0.8064),
 (Emb[ducking|VERB], 0.7877),
 (Emb[sprint|VERB], 0.7653),
 (Emb[scoot|VERB], 0.7647),
 (Emb[dart|VERB], 0.7621),
 (Emb[jump|VERB], 0.7528),
 (Emb[peek|VERB], 0.7518),
 (Emb[ducked|VERB], 0.7504),
 (Emb[bonk|VERB], 0.7495),
 (Emb[backflip|VERB], 0.746)]
```

We can also ask it to return an `EmbeddingSet` instead. That's what we're doing
below. We take our original embeddingset and we merge it with two more before
we visualise it.

```python
emb_bank_verb = lang.embset_similar("bank|VERB", n=10)
emb_bank_noun = lang.embset_similar("bank|NOUN", n=10)

(emb
 .merge(emb_bank_verb)
 .merge(emb_bank_noun)
 .transform(Pca(2))
 .plot_interactive("pca_0", "pca_1"))
```

<div id="sense2"></div>

<script>
fetch('sense2vec-2.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#sense2', out);
})
.catch(err => { throw err });
</script>

## Scikit-Learn 

Some of the languages inside of this package can be used in scikit-learn pipelines.
The spaCy and fasttext pipelines have compatible `.fit()` and `.transform()` methods
implemented. That means that you could write code like this: 

```python
import numpy as np 
from whatlies.language import SpacyLanguage
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("embed", SpacyLanguage("en_core_web_md")), 
    ("model", LogisticRegression())
])

X = [
    "i really like this post", 
    "thanks for that comment",
    "i enjoy this friendly forum",
    "this is a bad post", 
    "i dislike this article", 
    "this is not well written"
]

y = np.array([1, 1, 1, 0, 0, 0])

pipe.fit(X, y)
```

This pipeline is using the embeddings from spaCy now and passing those 
to the logistic regression.  

```
pipe.predict_proba(X)
# array([[0.37862409, 0.62137591],
#        [0.27858304, 0.72141696],
#        [0.21386529, 0.78613471],
#        [0.7155662 , 0.2844338 ],
#        [0.64924579, 0.35075421],
#        [0.76414156, 0.23585844]])
```
 
You could make a pipeline that generates both dense and sparse features by using a 
[FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html). 

```python
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

preprocess = FeatureUnion([
    ("dense", SpacyLanguage("en_core_web_md")),
    ("sparse", CountVectorizer())
])
```

### Caveats 

There's a few caveats to be aware of though. Fasttext as well as spaCy cannot be directly pickled
so that means that you won't be able to save a pipeline if there's a whatlies component
in it. It's also means that you cannot use a gridsearch. Where possible we try to 
test against scikit-learn's testing utilities but for now the usecases should be limited 
to exploration, not production.
