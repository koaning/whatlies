# Module whatlies.embedding


## Classes

### Embedding {: #Embedding }

```python
class Embedding(self, name, vector, orig=None)
```

This object represents a word embedding.

**Inputs**

- name: the name of the embedding
- vector: the numeric encoding of the embedding
- orig: the original name of the original embedding, is handled automatically

Initialize self.  See help(type(self)) for accurate signature.


------

#### Methods {: #Embedding-methods }

[**plot**](#Embedding.plot){: #Embedding.plot }

```python
def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False)
```

Handles the logic to perform a 2d plot in matplotlib.

**Input**

- kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
- color: the color to apply, only works for `scatter` and `arrow`
- xlabel: manually override the xlabel
- ylabel: manually override the ylabel
- show_operations: setting to also show the applied operations, only works for `text`
