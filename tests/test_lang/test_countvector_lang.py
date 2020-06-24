from whatlies.language import CountVectorLanguage


def test_basic_docs_usage1():
    lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
    embset = lang[["pizza", "pizzas", "firehouse", "firehydrant"]]
    assert embset.to_dataframe().shape == (4, 2)


def test_basic_docs_usage2():
    lang = CountVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
    lang.fit_manual(["pizza", "pizzas", "firehouse", "firehydrant", "cat", "dog"])
    embset = lang[["piza", "pizza", "pizzaz", "fyrehouse", "firehouse", "fyrehidrant"]]
    assert embset.to_dataframe().shape == (6, 2)
