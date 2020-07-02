# F.A.Q. 

### How do I save an interactive chart? 

The interactive charts that our library produces are made with [altair](https://altair-viz.github.io/). 
These charts use javascript for the interactivity and they are based on [vega](https://vega.github.io/vega-lite/).
You can represent the entire chart (including the data) as a json object. This means that you can always 
save a visluatisation as an html page or as a json file. 

```python
from whatlies.language import SpacyLanguage

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

lang = SpacyLanguage("en_core_web_md")
emb = lang[words]

p = emb.plot_interactive('man', 'woman')
p.to_html("plot.html")
p.to_json("plot.json")
```

A tutorial on how this works exactly can be found [here](https://calmcode.io/altair/json.html).

### How do I save an interactive chart for publication? 

You can also choose to save an interactive chart as an svg/png/pdf if you're interested 
in using an [altair](https://altair-viz.github.io/) visualisation in a publication. More
details are listed on their [documentation page](https://altair-viz.github.io/user_guide/saving_charts.html?highlight=save%20svg#png-svg-and-pdf-format)
in short you'll need to install the `altair_saver` package for this functionality.

To get this code to work you [may](https://github.com/RasaHQ/whatlies/issues/58) need to install some node
dependencies though. To install them locally in your project run;

```
npm install vega-lite vega-cli canvas
```

Once these are all installed, the following code snippet will work; 

```python
from whatlies.language import SpacyLanguage
from altair_saver import save

words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
         "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
         "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
         "person", "family", "brother", "sister"]

lang = SpacyLanguage("en_core_web_md")
emb = lang[words]

p = emb.plot_interactive('man', 'woman')
save(p, "chart.png")
```

This saves the following chart on disk;

![](images/chart.png)
