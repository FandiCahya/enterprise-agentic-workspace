from src.main import resolve_embedding_model


def test_default_embedding_model_is_supported():
    assert resolve_embedding_model() in [
        "models/text-embedding-004",
        "models/embedding-001",
        "models/gemini-embedding-001",
    ]
