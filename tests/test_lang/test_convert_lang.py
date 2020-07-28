import pytest

from whatlies.language import ConveRTLanguage


@pytest.fixture
def lang(request):
    lang = ConveRTLanguage(**request.param)
    return lang


@pytest.mark.parametrize(
    "lang, expected_shape",
    [
        ({"model_id": "convert", "signature": "default"}, (1024,)),
        ({"model_id": "convert", "signature": "encode_context"}, (512,)),
        ({"model_id": "convert", "signature": "encode_response"}, (512,)),
        ({"model_id": "convert", "signature": "encode_sequence"}, (512,)),
        ({"model_id": "convert-multi-context", "signature": "default"}, (1024,)),
        pytest.param(
            {"model_id": "convert-multi-context", "signature": "encode_context"},
            (512,),
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        ({"model_id": "convert-multi-context", "signature": "encode_response"}, (512,)),
        ({"model_id": "convert-multi-context", "signature": "encode_sequence"}, (512,)),
        ({"model_id": "convert-ubuntu", "signature": "default"}, (1024,)),
        pytest.param(
            {"model_id": "convert-ubuntu", "signature": "encode_context"},
            (512,),
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        ({"model_id": "convert-ubuntu", "signature": "encode_response"}, (512,)),
        ({"model_id": "convert-ubuntu", "signature": "encode_sequence"}, (512,)),
    ],
    indirect=["lang"],
)
def test_basic_usage(lang, expected_shape):
    embset = lang[["bank", "money on the bank", "bank of the river"]]
    assert len(embset) == 3
    assert embset["bank"].vector.shape == expected_shape


@pytest.mark.parametrize(
    "lang",
    [
        pytest.param(
            {"model_id": "convert-context", "signature": "encode_context"},
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        pytest.param(
            {"model_id": "convert", "signature": "encode"},
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
        pytest.param(
            {"model_id": "multi-convert", "signature": "encoded_context"},
            marks=pytest.mark.xfail(raises=ValueError, strict=True),
        ),
    ],
    indirect=["lang"],
)
def test_invalid_argument_values_raise_error(lang):
    pass
