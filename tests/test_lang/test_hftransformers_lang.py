import pytest
import numpy as np
from transformers import AutoTokenizer

from whatlies.language import HFTransformersLanguage


def load_model_and_tokenizer(model_name, tensor_type):
    if tensor_type == "tf":
        from transformers import TFAutoModel as AutoModel
    elif tensor_type == "pt":
        from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.mark.parametrize(
    "model_name, expected_shape",
    [
        ("sshleifer/tiny-gpt2", (2,)),
        ("sshleifer/tiny-distilroberta-base", (2,)),
    ],
)
@pytest.mark.parametrize("tensor_type", ["tf", "pt"])
def test_basic_usage_and_generated_embeddings(model_name, expected_shape, tensor_type):
    sentences = [
        "hello how are you?",
        "I'm fine, thanks!",
        "how about you?",
    ]

    lang = HFTransformersLanguage(model_name, framework=tensor_type)
    # Test for one single query
    emb = lang[sentences[0]]
    assert emb.vector.shape == expected_shape
    assert emb.name == sentences[0]

    model, tokenizer = load_model_and_tokenizer(model_name, tensor_type)
    tokens = tokenizer(
        sentences,
        padding=True,
        return_special_tokens_mask=True,
        return_tensors=tensor_type,
    )
    mask = tokens["special_tokens_mask"].numpy()
    del tokens["special_tokens_mask"]
    if tensor_type == "tf":
        output = model(tokens)
    elif tensor_type == "pt":
        output = model(**tokens)

    # Test for multiple queries
    emb = lang[sentences]
    assert len(emb) == len(sentences)
    for i, s in enumerate(sentences):
        if tensor_type == "tf":
            feats = output[0][i][np.logical_not(mask[i])].numpy()
        elif tensor_type == "pt":
            feats = output[0][i][np.logical_not(mask[i])].detach().numpy()
        assert emb[s].vector.shape == expected_shape
        assert np.allclose(emb[s].vector, feats.sum(axis=0))
