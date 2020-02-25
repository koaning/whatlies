# Module whatlies.common


## Functions

### handle_2d_plot {: #handle_2d_plot }

```python
def handle_2d_plot(embedding, kind, color=None, xlabel=None, ylabel=None, show_operations=False)
```

Handles the logic to perform a 2d plot in matplotlib.

**Input**
- embedding: a `whatlies.Embedding` object to plot
- kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
- color: the color to apply, only works for `scatter` and `arrow`
- xlabel: manually override the xlabel
- ylabel: manually override the ylabel
- show_operations: setting to also show the applied operations, only works for `text`
