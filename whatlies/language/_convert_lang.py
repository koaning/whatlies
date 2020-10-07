class ConveRTLanguage:
    """
    Important:
        This model has been deprecated. The original authors took the embeddings down.
    """

    MODEL_URL = {
        "convert": "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model.tar.gz",
        "convert-multi-context": "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model_multicontext.tar.gz",
        "convert-ubuntu": "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model_ubuntu.tar.gz",
    }

    MODEL_SIGNATURES = [
        "default",
        "encode_context",
        "encode_response",
        "encode_sequence",
    ]

    def __init__(self, model_id: str = "convert", signature: str = "default") -> None:
        pass

    def __getitem__(self, item):
        raise DeprecationWarning(
            "This model has been deprecated. The original authors took the embeddings down."
        )
