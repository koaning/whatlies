There's a few things that would be nice to have.

**Multiple Plot Metrics**

At the moment we only project onto axes to get x/y coordinates.
It might make sense to show the cosine distance to these
axes instead.

**Difference Charts**

It might be nice to show where a point was *before* a transformation
and then show where it is *after*. A single plot might make this nice.

**Multiple EmbeddingSets, Single Chart**

It would be especially cool if you could specify the color that way.

**Languages**

- language backends for huggingface models
- language backends for gensim models

**Stateful Transformations**

Let's say we have two embeddingsets; `emb1` and `emb2`. I might want
to transform `emb1` using `umap`. Currently the `umap` is stateless in
the sense that it fits and immediately transforms. Maybe we'd like it
to be stateful instead.

**Testing**

- it would be nice to have a good way of testing the charts
- it would be nice to be able to test multiple models without
having to download gigabytes in out CI
