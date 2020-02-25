# Module whatlies.language


## Classes

### SpacyLanguage {: #SpacyLanguage }

```python
class SpacyLanguage(self, model_name)
```

This object is used to lazily fetch `Embedding`s from a spaCy language
backend. Note that it is different than an `EmbeddingSet` in the sense
it does not have anything precomputed.

**Usage**:
```
lang = SpacyLanguage("en_core_web_md")
lang['python']
lang = SpacyLanguage("en_trf_robertabase_lg")
lang['programming in [python]']
```

Initialize self.  See help(type(self)) for accurate signature.
