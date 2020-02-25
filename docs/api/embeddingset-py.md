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


------

[**plot**](#EmbeddingSet.plot){: #EmbeddingSet.plot }

```python
def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False, **kwargs)
```
