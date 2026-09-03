from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_is_explicitly_pending() -> None:
    response = client.get("/api/v1/integration-readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert "PENDING TEAM INPUT" in body["message"]
    assert len(body["components"]) == 4


def test_prediction_does_not_run_without_model_contract() -> None:
    response = client.post("/api/v1/model/predict", json={"features": {}})
    assert response.status_code == 501
    assert "PENDING TEAM INPUT" in response.json()["detail"]["message"]