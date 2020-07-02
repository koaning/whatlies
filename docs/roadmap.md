There's a few things that would be nice to have. Feel free to start
a discussion on these topics in github.

**Multiple Plot Metrics**

At the moment we only project onto axes to get x/y coordinates.
It might make sense to show the cosine distance to these
axes instead.

And if we're allowing cosine distance ... we might allow for
flexible distance metrics in general.

**Table Summaries**

We've got a focus on charts now, but one imagines that calculating
tables with summary statistics is also relevant.

**Languages**

It would be nice to have other language backends, given that we do not
download all backends. We want this package to be light and users should
download manually.

- language backends for huggingface models

**Testing**

- it would be nice to have a good way of testing the charts
- it would be nice to be able to test multiple models without
having to download gigabytes into github actions

