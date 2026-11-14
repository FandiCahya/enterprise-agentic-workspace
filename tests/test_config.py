import importlib


def test_database_url_from_components(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "mydb")

    import src.core.config as config
    importlib.reload(config)

    assert config.get_database_url() == "postgresql+psycopg://u:p@db:5433/mydb"


def test_database_url_normalization(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/dbname")
    import src.core.config as config
    importlib.reload(config)

    assert config.get_database_url() == "postgresql+psycopg://user:pass@host:5432/dbname"


def test_cors_origins_parsing(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com, https://b.com")
    import src.core.config as config
    importlib.reload(config)

    assert config.get_cors_origins() == ["https://a.com", "https://b.com"]


def test_resolve_llm_model_prefers_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-test-model")
    import src.core.config as config
    importlib.reload(config)

    assert config.resolve_llm_model() == "gemini-test-model"
