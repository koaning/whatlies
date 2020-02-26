# Module whatlies.embeddingset


## Classes

### EmbeddingSet {: #EmbeddingSet }

```python
class EmbeddingSet(self, *embeddings, operations=None)
```

This object represents a set of `Embedding`s. You can use the same operations
as an `Embedding` but here we apply it to the entire set instead of a single
`Embedding`.

Initialize self.  See help(type(self)) for accurate signature.


------

#### Methods {: #EmbeddingSet-methods }

[**operate**](#EmbeddingSet.operate){: #EmbeddingSet.operate }

```python
def operate(self, other, operation)
```

Attaches an operation to perform on the `EmbeddingSet`.

**Inputs**

- other: the other `Embedding`
- operation: the operation to apply to all embeddings in the set, can be `+`, `-`, `|`, `>>`, `>`

**Output**

A new `EmbeddingSet`

------

[**plot**](#EmbeddingSet.plot){: #EmbeddingSet.plot }

```python
def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False, **kwargs)
```

Handles the logic to perform a 2d plot in matplotlib.

**Input**

- kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
- x_axis: what embedding to use as a x-axis
- y_axis: what embedding to us as a y-axis
- color: the color to apply, only works for `scatter` and `arrow`
- xlabel: manually override the xlabel
- ylabel: manually override the ylabel
- show_operations: setting to also show the applied operations, only works for `text`
