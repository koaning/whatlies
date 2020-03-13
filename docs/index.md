# whatlies 

<img src="logo.png">

A library that tries help you to understand. "What lies in word embeddings?"

This small library offers tools to make visualisation easier of both
word embeddings as well as operations on them. This should be considered
an alpha project.

This library will allow you to make visualisations of transformations
of word embeddings. Think of stuff like;

<script src="https://cdn.jsdelivr.net/npm/vega@5.10.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@4.6.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.3.2"></script> 
<div id="vis"></div>

<script>
var spec = {
  "config": {"view": {"continuousWidth": 400, "continuousHeight": 300}},
  "hconcat": [
    {
      "layer": [
        {
          "mark": {"type": "circle", "size": 60},
          "encoding": {
            "tooltip": [
              {"type": "nominal", "field": "name"},
              {"type": "nominal", "field": "original"}
            ],
            "x": {"type": "quantitative", "field": "pca_0"},
            "y": {"type": "quantitative", "field": "pca_1"}
          },
          "selection": {
            "selector009": {
              "type": "interval",
              "bind": "scales",
              "encodings": ["x", "y"]
            }
          },
          "title": "pca_0 vs. pca_1"
        },
        {
          "mark": {"type": "text", "color": "black", "dx": -15, "dy": 3},
          "encoding": {
            "text": {"type": "nominal", "field": "original"},
            "x": {"type": "quantitative", "field": "pca_0"},
            "y": {"type": "quantitative", "field": "pca_1"}
          }
        }
      ],
      "data": {"name": "data-c65a008618c7155525ca27ca465c1972"}
    },
    {
      "layer": [
        {
          "mark": {"type": "circle", "size": 60},
          "encoding": {
            "tooltip": [
              {"type": "nominal", "field": "name"},
              {"type": "nominal", "field": "original"}
            ],
            "x": {"type": "quantitative", "field": "umap_0"},
            "y": {"type": "quantitative", "field": "umap_1"}
          },
          "selection": {
            "selector010": {
              "type": "interval",
              "bind": "scales",
              "encodings": ["x", "y"]
            }
          },
          "title": "umap_0 vs. umap_1"
        },
        {
          "mark": {"type": "text", "color": "black", "dx": -15, "dy": 3},
          "encoding": {
            "text": {"type": "nominal", "field": "original"},
            "x": {"type": "quantitative", "field": "umap_0"},
            "y": {"type": "quantitative", "field": "umap_1"}
          }
        }
      ],
      "data": {"name": "data-c9d79ce4e8383197845923751e4169ab"}
    }
  ],
  "$schema": "https://vega.github.io/schema/vega-lite/v4.0.2.json",
  "datasets": {
    "data-c65a008618c7155525ca27ca465c1972": [
      {
        "pca_0": -1.2535821199417114,
        "pca_1": 3.586676597595215,
        "name": "prince",
        "original": "prince"
      },
      {
        "pca_0": -0.9741005897521973,
        "pca_1": 3.7290642261505127,
        "name": "princess",
        "original": "princess"
      },
      {
        "pca_0": -1.0501278638839722,
        "pca_1": -2.0268447399139404,
        "name": "nurse",
        "original": "nurse"
      },
      {
        "pca_0": -0.9151343703269958,
        "pca_1": -2.0640790462493896,
        "name": "doctor",
        "original": "doctor"
      },
      {
        "pca_0": 0.8430215716362,
        "pca_1": -0.06272825598716736,
        "name": "banker",
        "original": "banker"
      },
      {
        "pca_0": -0.43132147192955017,
        "pca_1": -1.5975831747055054,
        "name": "man",
        "original": "man"
      },
      {
        "pca_0": -1.073258876800537,
        "pca_1": -1.6417843103408813,
        "name": "woman",
        "original": "woman"
      },
      {
        "pca_0": -3.006711006164551,
        "pca_1": 0.08844566345214844,
        "name": "cousin",
        "original": "cousin"
      },
      {
        "pca_0": -3.256138563156128,
        "pca_1": -0.2640823423862457,
        "name": "neice",
        "original": "neice"
      },
      {
        "pca_0": -0.8844128251075745,
        "pca_1": 4.284396648406982,
        "name": "king",
        "original": "king"
      },
      {
        "pca_0": -0.9637596607208252,
        "pca_1": 3.8633878231048584,
        "name": "queen",
        "original": "queen"
      },
      {
        "pca_0": -0.37136709690093994,
        "pca_1": -1.4314926862716675,
        "name": "dude",
        "original": "dude"
      },
      {
        "pca_0": -0.33022043108940125,
        "pca_1": -2.0039174556732178,
        "name": "guy",
        "original": "guy"
      },
      {
        "pca_0": 0.005235184449702501,
        "pca_1": -1.1506175994873047,
        "name": "gal",
        "original": "gal"
      },
      {
        "pca_0": 2.1890616416931152,
        "pca_1": -0.3874746263027191,
        "name": "fire",
        "original": "fire"
      },
      {
        "pca_0": 0.8451134562492371,
        "pca_1": -1.5011231899261475,
        "name": "dog",
        "original": "dog"
      },
      {
        "pca_0": 0.9290971755981445,
        "pca_1": -0.6315348148345947,
        "name": "cat",
        "original": "cat"
      },
      {
        "pca_0": 1.9717415571212769,
        "pca_1": 0.1653272807598114,
        "name": "mouse",
        "original": "mouse"
      },
      {
        "pca_0": 3.6464710235595703,
        "pca_1": 0.8340242505073547,
        "name": "red",
        "original": "red"
      },
      {
        "pca_0": 0.6217125654220581,
        "pca_1": -0.011615407653152943,
        "name": "bluee",
        "original": "bluee"
      },
      {
        "pca_0": 4.070711135864258,
        "pca_1": 0.8428587913513184,
        "name": "green",
        "original": "green"
      },
      {
        "pca_0": 3.9739019870758057,
        "pca_1": 0.8420842289924622,
        "name": "yellow",
        "original": "yellow"
      },
      {
        "pca_0": 3.1992409229278564,
        "pca_1": -0.76308274269104,
        "name": "water",
        "original": "water"
      },
      {
        "pca_0": -0.4370383024215698,
        "pca_1": -1.8055583238601685,
        "name": "person",
        "original": "person"
      },
      {
        "pca_0": -1.5232115983963013,
        "pca_1": -0.6766523718833923,
        "name": "family",
        "original": "family"
      },
      {
        "pca_0": -2.889375925064087,
        "pca_1": -0.0504235103726387,
        "name": "brother",
        "original": "brother"
      },
      {
        "pca_0": -2.935547351837158,
        "pca_1": -0.16567137837409973,
        "name": "sister",
        "original": "sister"
      }
    ],
    "data-c9d79ce4e8383197845923751e4169ab": [
      {
        "umap_0": 7.683071136474609,
        "umap_1": 6.8826069831848145,
        "name": "prince",
        "original": "prince"
      },
      {
        "umap_0": 8.119927406311035,
        "umap_1": 6.849494457244873,
        "name": "princess",
        "original": "princess"
      },
      {
        "umap_0": 7.300721645355225,
        "umap_1": 4.110742092132568,
        "name": "nurse",
        "original": "nurse"
      },
      {
        "umap_0": 7.764898777008057,
        "umap_1": 3.85836124420166,
        "name": "doctor",
        "original": "doctor"
      },
      {
        "umap_0": 8.261445045471191,
        "umap_1": 5.063197612762451,
        "name": "banker",
        "original": "banker"
      },
      {
        "umap_0": 8.816865921020508,
        "umap_1": 4.588306427001953,
        "name": "man",
        "original": "man"
      },
      {
        "umap_0": 7.818793296813965,
        "umap_1": 4.510528564453125,
        "name": "woman",
        "original": "woman"
      },
      {
        "umap_0": 7.104825496673584,
        "umap_1": 5.862237930297852,
        "name": "cousin",
        "original": "cousin"
      },
      {
        "umap_0": 6.475111484527588,
        "umap_1": 5.630688190460205,
        "name": "neice",
        "original": "neice"
      },
      {
        "umap_0": 7.501045227050781,
        "umap_1": 6.528558254241943,
        "name": "king",
        "original": "king"
      },
      {
        "umap_0": 7.875270843505859,
        "umap_1": 6.347613334655762,
        "name": "queen",
        "original": "queen"
      },
      {
        "umap_0": 8.438183784484863,
        "umap_1": 3.872556686401367,
        "name": "dude",
        "original": "dude"
      },
      {
        "umap_0": 8.448274612426758,
        "umap_1": 4.3103251457214355,
        "name": "guy",
        "original": "guy"
      },
      {
        "umap_0": 7.9519147872924805,
        "umap_1": 5.135070323944092,
        "name": "gal",
        "original": "gal"
      },
      {
        "umap_0": 8.944172859191895,
        "umap_1": 5.7392354011535645,
        "name": "fire",
        "original": "fire"
      },
      {
        "umap_0": 9.735146522521973,
        "umap_1": 4.4750895500183105,
        "name": "dog",
        "original": "dog"
      },
      {
        "umap_0": 9.864075660705566,
        "umap_1": 4.957643985748291,
        "name": "cat",
        "original": "cat"
      },
      {
        "umap_0": 9.328947067260742,
        "umap_1": 5.098883628845215,
        "name": "mouse",
        "original": "mouse"
      },
      {
        "umap_0": 10.150426864624023,
        "umap_1": 6.103381633758545,
        "name": "red",
        "original": "red"
      },
      {
        "umap_0": 8.376856803894043,
        "umap_1": 5.627901554107666,
        "name": "bluee",
        "original": "bluee"
      },
      {
        "umap_0": 9.726550102233887,
        "umap_1": 6.283388614654541,
        "name": "green",
        "original": "green"
      },
      {
        "umap_0": 9.834991455078125,
        "umap_1": 5.793656349182129,
        "name": "yellow",
        "original": "yellow"
      },
      {
        "umap_0": 9.267016410827637,
        "umap_1": 6.091810703277588,
        "name": "water",
        "original": "water"
      },
      {
        "umap_0": 8.906523704528809,
        "umap_1": 4.007486820220947,
        "name": "person",
        "original": "person"
      },
      {
        "umap_0": 6.7370147705078125,
        "umap_1": 4.977278709411621,
        "name": "family",
        "original": "family"
      },
      {
        "umap_0": 7.2880024909973145,
        "umap_1": 5.024892807006836,
        "name": "brother",
        "original": "brother"
      },
      {
        "umap_0": 7.031651496887207,
        "umap_1": 5.402203559875488,
        "name": "sister",
        "original": "sister"
      }
    ]
  }
}

vegaEmbed('#vis', spec);
</script>

## Installation 

For now we allow for installation with pip but only via git.

```bash
pip install git+git@github.com:RasaHQ/whatlies.git
```

## Local Development

If you want to develop locally you can start by running this command. 

```bash
make develop
```

### Documentation 

This is generated via

```
make docs
```

## Produced 

This project was initiated at [![](rasa.png)](https://rasa.com) 