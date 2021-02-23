import pathlib
import warnings

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin

from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.shared.nlu.training_data.message import Message


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return RasaNLUInterpreter(nlu_model)


class DIETLanguage(SklearnTransformerMixin):
    """
    **Usage**:

    ```python
    from whatlies.language import DIETLangauge
    lang = TFIDFVectorLanguage(n_components=2, ngram_range=(1, 2), analyzer="char")
    lang[['pizza', 'pizzas', 'firehouse', 'firehydrant']]
    ```
    """

    def __init__(self, model_path):
        self.model_path = model_path
        folder, file = (
            pathlib.Path(model_path).parent,
            pathlib.Path(model_path).parts[-1],
        )
        interpreter = load_interpreter(model_dir=folder, model=file)
        self.pipeline = interpreter.interpreter.pipeline

    def __getitem__(self, item):
        """
        Retreive a single embedding or a set of embeddings. We retreive the sentence encoding that
        belongs to the entire utterance.

        Arguments:
            item: single string or list of strings

        **Usage**
        ```python
        > lang = BytePairLanguage(lang="en")
        > lang['python']
        > lang[['python', 'snake']]
        > lang[['nobody expects', 'the spanish inquisition']]
        ```
        """
        if isinstance(item, str):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                msg = Message({"text": item})
                for p in self.pipeline:
                    p.process(msg)
                diagnostic_data = msg.as_dict_nlu()["diagnostic_data"]
                key_of_interest = [k for k in diagnostic_data.keys() if "DIET" in k][0]
                # It's assumed that the final token in the array here represents the __CLS__ token.
                # These are also known as the "sentence embeddings"
                tensors = diagnostic_data[key_of_interest]["text_transformed"]
                return Embedding(item, tensors[-1][-1])
        if isinstance(item, list):
            return EmbeddingSet(*[self[i] for i in item])
        raise ValueError(f"Item must be list of strings got {item}.")
