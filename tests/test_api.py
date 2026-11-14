from fastapi.testclient import TestClient

import src.main as main


client = TestClient(main.app)


def test_health_check_success(monkeypatch):
    monkeypatch.setattr(main, "_check_db", lambda: None)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "online"
    assert body["database"] == "connected"


def test_ingest_validation_empty_payload():
    response = client.post("/api/v1/documents/ingest", json={})
    assert response.status_code == 422


def test_search_validation_empty_query():
    response = client.post("/api/v1/documents/search", json={"query": "", "top_k": 3})
    assert response.status_code == 422


def test_api_key_protection(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")
    # reload config-derived getter behavior by patching get_api_key indirection in main
    monkeypatch.setattr(main, "get_api_key", lambda: "secret-key")

    resp = client.post(
        "/api/v1/documents/search",
        json={"query": "test", "top_k": 3},
    )
    assert resp.status_code == 401

    # with key, endpoint proceeds to handler (mock vector store to avoid DB)
    class DummyVS:
        def similarity_search(self, query, k=3):
            return []

    monkeypatch.setattr(main, "get_vector_store", lambda: DummyVS())
    resp_ok = client.post(
        "/api/v1/documents/search",
        json={"query": "test", "top_k": 3},
        headers={"x-api-key": "secret-key"},
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["results_count"] == 0
