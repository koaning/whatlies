from operator import add, sub, rshift, or_

import pandas as pd
import altair as alt

from whatlies.common import plot_graph_layout


class EmbeddingSet:
    """
    This object represents a set of `Embedding`s. You can use the same operations
    as an `Embedding` but here we apply it to the entire set instead of a single
    `Embedding`.

    **Parameters**

    - **embeddings**: list of embeddings or dictionary with name: embedding pairs
    - **operations**: deprecated
    - **name**: custom name of embeddingset

    Usage:

    ```
    from whatlies.embedding import Embedding
    from whatlies.embeddingset import EmbeddingSet
    ```
    """
    def __init__(self, *embeddings, operations=None, name=None):
        if not name:
            name = 'EmbSet'
        self.name = name
        if len(embeddings) == 1:
            # we assume it is a dictionary here
            self.embeddings = embeddings[0]
        else:
            # we assume it is a tuple of tokens
            self.embeddings = {t.name: t for t in embeddings}
        self.operations = [] if not operations else operations

    def operate(self, other, operation):
        """
        Attaches an operation to perform on the `EmbeddingSet`.

        **Inputs**

        - other: the other `Embedding`
        - operation: the operation to apply to all embeddings in the set, can be `+`, `-`, `|`, `>>`, `>`

        **Output**

        A new `EmbeddingSet`
        """
        new_embeddings = {k: operation(emb, other) for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, operations=self.operations + [(other, operation)])

    def __add__(self, other):
        return self.operate(other, add)

    def __sub__(self, other):
        return self.operate(other, sub)

    def __or__(self, other):
        return self.operate(other, or_)

    def __rshift__(self, other):
        return self.operate(other, rshift)

    def compare_against(self, other, mapping='direct'):
        if mapping == 'direct':
            return [v > other for k, v in self.embeddings.items()]

    def transform(self, transformer):
        return transformer(self)

    def __getitem__(self, thing):
        if not isinstance(thing, list):
            return self.embeddings[thing]
        new_embeddings = {k: emb for k, emb in self.embeddings.items()}
        names = ','.join(thing)
        return EmbeddingSet(new_embeddings, name=f"{self.name}.subset({names})")

    def __repr__(self):
        result = self.name
        translator = {add: '+', sub: '-', or_: '|', rshift: '>>'}
        for tok, op in self.operations:
            result = f"({result} {translator[op]} {tok.name})"
        return result

    def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False, **kwargs):
        """
        Handles the logic to perform a 2d plot in matplotlib.

        **Input**

        - kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
        - x_axis: what embedding to use as a x-axis
        - y_axis: what embedding to us as a y-axis
        - color: the color to apply, only works for `scatter` and `arrow`
        - xlabel: manually override the xlabel
        - ylabel: manually override the ylabel
        - show_operations: setting to also show the applied operations, only works for `text`
        """
        for k, token in self.embeddings.items():
            token.plot(kind=kind, x_axis=x_axis, y_axis=y_axis, color=color, show_operations=show_operations, **kwargs)
        return self

    def plot_graph_layout(self, kind='cosine', **kwargs):
        """
        Handles the logic to plot a 2d graph using cosine distance
        :return:
        """
        plot_graph_layout(self.embeddings, kind, **kwargs)
        return self

    def plot_interactive(self, x_axis, y_axis, annot=True, show_axis_point=False):
        if isinstance(x_axis, str):
            x_axis = self[x_axis]
        if isinstance(y_axis, str):
            y_axis = self[y_axis]

        plot_df = pd.DataFrame({
            'x_axis': self.compare_against(x_axis),
            'y_axis': self.compare_against(y_axis),
            'name': [v.name for v in self.embeddings.values()],
            'original': [v.orig for v in self.embeddings.values()]
        })

        if not show_axis_point:
            plot_df = plot_df.loc[lambda d: ~d['name'].isin([x_axis.name, y_axis.name])]

        result = alt.Chart(plot_df).mark_circle(size=60).encode(
            x=alt.X('x_axis', axis=alt.Axis(title=x_axis.name)),
            y=alt.X('y_axis', axis=alt.Axis(title=y_axis.name)),
            tooltip=['name', 'original'],
        ).properties(title=f"{x_axis.name} vs. {y_axis.name}").interactive()

        if annot:
            text = alt.Chart(plot_df).mark_text(dx=-15, dy=3, color='black').encode(
                x=alt.X('x_axis', axis=alt.Axis(title=x_axis.name)),
                y=alt.X('y_axis', axis=alt.Axis(title=y_axis.name)),
                text='original'
            )
            result = result + text
        return result

    def plot_interactive_matrix(self, *axes, annot=True, show_axis_point=False, width=200, height=200):
        plot_df = pd.DataFrame({ax: self.compare_against(self[ax]) for ax in axes})
        plot_df['name'] = [v.name for v in self.embeddings.values()]
        plot_df['original'] = [v.orig for v in self.embeddings.values()]

        if not show_axis_point:
            plot_df = plot_df.loc[lambda d: ~d['name'].isin(axes)]

        result = alt.Chart(plot_df).mark_circle().encode(
            x=alt.X(alt.repeat("column"), type='quantitative'),
            y=alt.Y(alt.repeat("row"), type='quantitative'),
            tooltip=['name', 'original'],
            text='original',
        )
        if annot:
            text_stuff = result.mark_text(dx=-15, dy=3, color='black').encode(
                x=alt.X(alt.repeat("column"), type='quantitative'),
                y=alt.Y(alt.repeat("row"), type='quantitative'),
                tooltip=['name', 'original'],
                text='original',
            )
            result = result + text_stuff

        result = result.properties(
            width=width,
            height=height
        ).repeat(
            row=axes[::-1],
            column=axes
        ).interactive()

        return result