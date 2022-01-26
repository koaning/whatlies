from whatlies import Embedding, EmbeddingSet


def reverse_strings(embset):
    """
    This helper will reverse the strings in the embeddingset. This can be useful
    for making matplotlib plots with Arabic texts.

    This helper is meant to be used via `EmbeddingSet.pipe()`.

    Arguments:
        embset: EmbeddingSet to adapt

    Usage:

    ```python
    from whatlies.helpers import reverse_strings
    from whatlies.language import BytePairLanguage

    translation = {
       "man":"رجل",
       "woman":"امرأة",
       "king":"ملك",
       "queen":"ملكة",
       "brother":"أخ",
       "sister":"أخت",
       "cat":"قطة",
       "dog":"كلب",
       "lion":"أسد",
       "puppy":"جرو",
       "male student":"طالب",
       "female student":"طالبة",
       "university":"جامعة",
       "school":"مدرسة",
       "kitten":" قطة صغيرة",
       "apple" : "تفاحة",
       "orange" : "برتقال",
       "cabbage" : "كرنب",
       "carrot" : "جزرة"
    }

    lang_cv  = BytePairLanguage("ar")

    arabic_words = list(words.values())

    # before
    lang_cv[translation].plot_similarity()

    # after
    lang_cv[translation].pipe(reverse_strings).plot_similarity()
    ```

    ![](https://koaning.github.io/whatlies/images/arabic-before-after.png)
    """
    return EmbeddingSet(
        *[Embedding(name=e.name[::-1], vector=e.vector) for e in embset]
    )
