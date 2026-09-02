from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_check():
    """Pastikan server dan koneksi PGVector merespons 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_ingest_empty_validation():
    """Pastikan validasi skema request body berjalan saat payload kosong."""
    response = client.post("/api/v1/documents/ingest", json={})
    assert response.status_code == 422  # Unprocessable Entity
