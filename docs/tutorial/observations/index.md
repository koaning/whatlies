<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script>

# Observations 

This document will demonstrate some observations of embeddings. The goal of this document
is to two-fold. 

1. It hopes to explain how to use `whatlies`. 
2. It hopes to explain interesting elements of different embeddings. 

## Spelling Errors 

Especially when you're designing a chatbot, spelling errors occur all the time. So 
how do different embeddings deal with this? Let's compare three different language
backends here. 

```python
from whatlies.language import FasttextLanguage, SpacyLanguage, CountVectorLanguage

lang_spacy = SpacyLanguage("en_core_web_md")
lang_fasttext = FasttextLanguage("cc.en.300.bin")
lang_cv = CountVectorLanguage(n_components=2, ngram_range=(1, 3), analyzer="char")
lang_cv.fit_manual(['pizza', 'pizzas', 'firehouse', 'firehydrant', 
                    'cat', 'dog', 'pikachu', 'pokemon'])
```

Besides fetching a fasttext and spaCy model you'll notice that we're 
also manually fitting the `CountVectorLanguage`. Take a note
of this because this fact will be meaningful later.

```python
words = ['piza', 'pizza', 'pizzza', 'italian', 'sushi', 'japan', 'burger', 
         'fyrehouse', 'firehouse', 'fyrehidrant',
         'house', 'tree', 'elephant', 'pikachu', 'pokemon']

def mk_plot(lang):
    return (lang[words]
            .transform(Pca(2))
            .plot_interactive()
            .properties(height=250, width=250, title=lang.__class__.__name__))

(mk_plot(lang_spacy) & mk_plot(lang_fasttext) & mk_plot(lang_cv))
```

<div id="c1"></div>

<script>
fetch('chart-1.json')
.then(res => res.json())
.then((out) => {
  vegaEmbed('#c1', out);
})
.catch(err => { throw err });
</script>

You may notice a few things here. 

- The `CountVectorLanguage` only looks at character similarities which it is why
it feels that "piza", "pizza" and "pizzza" are all very similar. Notice how spaCy
disagrees with this. You may also note that fasttext is "in between". That is because
these embeddings also encode the subtokens. 
- The fasttext embeddings seem to do a better job at catching certain forms of meaning. 
It understands that pikachu, pokemon and japan are related while also associating
burger/sushi with pizza. The spaCy model also captures this but the clustering is 
less appearant. This can be due to the dimensionality reduction though.
- You'll notice that the two clearly misspelled words, fyrehouse and fyrehidrant, 
are mapped to the same point in the spaCy embedding. When you check the embeddings
for both you'll confirm why. SpaCy maps a token to a vector of zeros is it is not 
available in the vocabulary. SpaCy may occasionally also map two different tokens 
to the same embedding in an attempt to save on disk space. Fasttext is able to recover 
more context because of the subtoken embeddings but also because the embeddings are 
**way bigger**. The fasttext embeddings unzipped can be 7GB on disk while spaCy 
model is on 115MB on disk. 
- Notice how the `CountVectorLanguage` has a cluster with lots of things in it. 
It clusters together "house", "elephant", "sushi" and "pokemon". This isn't because
of shared meaning. It because these words contain combinations of characters that
weren't there in the training set that we gave to the `fit_manual` method in the beginning. 
Note that "picachu" is similar to "pizza" for a similar reason.


