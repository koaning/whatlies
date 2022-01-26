## Scikit-Learn

Many of the language-backends inside of this package can be used in scikit-learn pipelines.
We've implemented a compatible `.fit()` and `.transform()` API which means that
you could write scikit-learn pipelines like this:

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
    ("sparse_word", CountVectorizer()),
    ("sparse_subword", CountVectorizer(analyzer="char", ngram_range=(2, 4)))
])
```

### Supported Models

Every language backend that this library offers is compatible for use in a scikit-learn
pipeline. This includes the following;

- `whatlies.language.SpacyLanguage`
- `whatlies.language.FasttextLanguage`
- `whatlies.language.CountVectorLanguage`
- `whatlies.language.BytePairLanguage`
- `whatlies.language.GensimLanguage`
- `whatlies.language.HFTransformersLanguage`
- `whatlies.language.TFHubLanguage`
- `whatlies.language.UniversalSentenceLanguage`
- `whatlies.language.SentenceTFMLanguage`
- `whatlies.language.UniversalSentenceLanguage`
- `whatlies.language.LaBSELanguage`

### Caveats

There's a few caveats to be aware of though. In general these language backends cannot be
directly pickled so that means that you won't be able to save a pipeline if there's a whatlies
component in it. This also means that you cannot use a gridsearch. Where possible we try to
test against scikit-learn's testing utilities but for now the use-case is limited for use in
a `Pipeline`. You should assume that you cannot use `GridSearchCV` and that you cannot pickle
to disk.

If you see a way to properly support this in general, let us know on github by
creating an [issue](https://github.com/koaning/whatlies/issues).
