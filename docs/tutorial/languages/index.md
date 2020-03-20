In this tool we have support for different language backends and
depending on the language backend you may get slightly different behavior.

## Multiple Tokens

## Bert Style 

## Sense to Vec

```python
from whatlies.language import Sense2VecLangauge
lang = Sense2VecLangauge("path/downloaded/s2v")

lang["bank|NOUN"]
```